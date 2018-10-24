import models.utils as ut
from functools import reduce

from fastFM import bpr
import numpy as np
from scipy.sparse import lil_matrix, hstack, eye
import scipy.io as sio

matfile = '../data/fpmc_dataset.mat'

mat = sio.loadmat(matfile, struct_as_record=False, squeeze_me=True)

trainSet = mat['fpmc_dataset']
testLast = mat['testLast']
testLOO = mat['testllo']
nUser = mat['nUser']
nItem = mat['nItem']

sap = 50
X_UserTrain = lil_matrix(((sap+1)*len(trainSet), nUser+1), dtype=np.float32)
X_ItemTrain = lil_matrix(((sap+1)*len(trainSet),nItem+1), dtype=np.float32)
X_TargetTrain = lil_matrix(((sap+1)*len(trainSet),nItem+1), dtype=np.float32)
Y_train = np.zeros([sap*len(trainSet),2], dtype=np.int32)
nTrain = len(trainSet)
for i, row in enumerate(trainSet):
    m = len(row.Session) if isinstance(row.Session, list) else 1
    begin = (sap+1)*i
    X_UserTrain[begin:begin+sap+1, row.User] = 1
    X_ItemTrain[begin:begin+sap+1, row.Session] = 1/m
    X_TargetTrain[begin, row.Target] = 1
    X_TargetTrain[np.arange(begin+1, begin+sap+1), np.random.randint(0, nItem, sap)] = 1
    beginY = sap * i
    Y_train[beginY:beginY+sap, 0] = begin
    Y_train[beginY:beginY+sap, 1] = np.arange(begin+1, begin+sap+1)
    if i % 1000 == 0:
        print("Constructing %d/%d"%(i, nTrain))

X_train = hstack([X_ItemTrain, X_TargetTrain]).tocsc()



# Build Model
print('Start training')
embed_len = 20
fm = bpr.FMRecommender(n_iter=5000000, init_stdev=0.1, rank=embed_len,
                       l2_reg_w=0, l2_reg_V=0, l2_reg=0, step_size=0.1)
fm.fit(X_train, Y_train)


print('Start evaluation')

X_Target = eye(nItem+1, dtype=np.float32)

atK = np.arange(5, 51, 5)
ranklist = []
nTest = len(testLast)
recs = []
for i, row in enumerate(testLast):
    # X_User = lil_matrix((nItem + 1, nUser + 1), dtype=np.float32)
    # X_User[:, row.User] = 1
    X_Test = lil_matrix((nItem + 1, nItem + 1), dtype=np.float32)
    X_Test[:, row.Session] = 1/len(row.Session) if isinstance(row.Session, list) else 1
    scores = fm.predict(hstack([X_Test, X_Target]))
    rank = (scores > scores[row.TestCase]).sum() + 1
    ranklist.append(rank)
    recs.append(np.argpartition(scores, -10)[-10:])
    if i % 1000 == 0:
        print("Evaluating %d/%d" % (i, nTest))

ranklist = np.array(ranklist)
print("AUC: %g" % (np.mean((nItem - ranklist) / nItem)))
print("MRR")
print(np.mean((1 / ranklist[:, None]) * (ranklist[:, None] <= atK), axis=0))
print("RECALL")
print(np.mean(ranklist[:, None] <= atK, axis=0))
print("Div: %g"%ut.diversity(recs))


ranklist = []
nTest = len(testLOO)

for i, row in enumerate(testLOO):
    # X_User = lil_matrix((nItem + 1, nUser + 1), dtype=np.float32)
    # X_User[:, row.User] = 1
    X_Test = lil_matrix((nItem + 1, nItem + 1), dtype=np.float32)
    X_Test[:, row.Session] = 1/len(row.Session) if isinstance(row.Session, list) else 1
    scores = fm.predict(hstack([X_Test, X_Target]))
    rank = (scores > scores[row.TestCase]).sum() + 1
    ranklist.append(rank)
    recs.append(np.argpartition(scores, -10)[-10:])
    if i % 1000 == 0:
        print("Evaluating %d/%d" % (i, nTest))

ranklist = np.array(ranklist)
print("AUC: %g" % (np.mean((nItem - ranklist) / nItem)))
print("MRR")
print(np.mean((1 / ranklist[:, None]) * (ranklist[:, None] <= atK), axis=0))
print("RECALL")
print(np.mean(ranklist[:, None] <= atK, axis=0))
print("Div: %g"%ut.diversity(recs))
