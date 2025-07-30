# %%
import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

### 데이터셋 불러오기 
def build_index(dataset_name):
    dataset_name = "All_Beauty"
    ui_mat = np.loadtxt(f"/Users/optim/Desktop/Recommendation/{dataset_name}.txt", dtype=np.int32)
    print("✅ Loaded dataset shape:", ui_mat.shape)
    print("🔍 First 10 rows:\n", ui_mat[:5])

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])
    return u2i_index, i2u_index


#%%
# sampler for batch generation -> 유저가 상호작용하지 않은 아이템을 찾기 위해서 반복함!!
def random_neq(l, r, s): # l과 r 사이 랜덤한 정수 반환한다. 유저가 관람하지 않은 영화 목록에서 랜덤으로 한개씩 뽑아서 pos와 동일한 사이즈로 구성한다. 
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t 

##### 2. input 값 만드는 과정 !!! ##### 
# n개의 시퀀스로 모두 통일해줘야함. 
def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid): # 샘플 함수 지정함. 

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1) # 만약 유저안의 값들이 0 또는 1일 경우, 결측치이므로 다시 시도
        
        seq = np.zeros([maxlen], dtype=np.int32) # 시퀀스 (input)
        pos = np.zeros([maxlen], dtype=np.int32) # positive ground truth로 사용할 것 
        neg = np.zeros([maxlen], dtype=np.int32) # negative 연관 안 된 것 
        nxt = user_train[uid][-1] # 학습 시퀀셜 n-2번째. 학습 시 n-2를 train으로 사용함.
        idx = maxlen - 1 # 인덱스는 1개 삭제함. 

        ts = set(user_train[uid]) # 유저의 아이템 시퀀스 값을 집합 형태로 = 중복값 제거함.
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt # 다음에 실제로 클릭한 아이템 = 실제 정답 
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))

#%%
class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

#%%
# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list) # 원소값을 업데이팅하는 딕셔너리 
    user_train = {} # 각각 데이터들을 넣는다. 
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(f"/Users/optim/Desktop/Recommendation/{fname}.txt", 'r')
    for line in f:
        u, i = line.rstrip().split(' ') #첫번째칸은 유저, 두번째칸은 아이템이다. 
        u = int(u)
        i = int(i)
        usernum = max(u, usernum) # 얼마까지 있는지 모르니까 하나씩 넣기 
        itemnum = max(i, itemnum)
        User[u].append(i) # 유저와 아이템에 대한 딕셔너리 생성. 유저1 : 아이템 리스트 
        
    # 기본적으로 이미 시간순으로 정렬되어 있는 것이라서 원래 순서만 보존하면 됨. 
    for user in User:
        nfeedback = len(User[user]) # 유저1 아이템 리스트  시퀀셜 리스트 -> 3보다 작을 경우
        if nfeedback < 3:
            user_train[user] = User[user] # train 데이터만 사용한다. 
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2] # 나머지의 경우 맨 마지막 2개 뺴고 넣기 
            user_valid[user] = []
            user_valid[user].append(User[user][-2]) # n-1번째 아이템을 valid
            user_test[user] = []
            user_test[user].append(User[user][-1]) # n번째 것을 test로 사용한다.
    return [user_train, user_valid, user_test, usernum, itemnum] 

# %%
# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]

        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

#%%
# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]

        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

# %%
u2i_index, i2u_index = build_index("")


# %%
build_index('/Users/optim/Desktop/Recommendation/All_Beauty.txt')