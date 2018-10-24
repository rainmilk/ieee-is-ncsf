import math
import scipy.io as sio
import numpy as np
import models.utils as ut

matfile = '../data/matlab.mat'

mat = sio.loadmat(matfile, struct_as_record=False, squeeze_me=True)
nb_item = mat['nItem']
session = mat['session']
testIdx = mat['testingIdx']
sessionSet = {r.User:[] for r in session}
testSet = {r.User:set() for r in session}
for r, t in zip(session, testIdx):
    ul = sessionSet[r.User]
    idx = len(ul)
    if t and idx > 0:
        testSet[r.User].add(idx)
    ul.append(r.Session)

nb_trn_spl = 0
nb_tst_spl = 0
nb_tst_spl_last = 0
for k,v in sessionSet.items():
    # if len(v) > 1:
        tl = testSet[k]
        for i in range(1, len(v)):
            if i in tl:
                nb_tst_spl_last += 1
                nb_tst_spl += len(v[i])
            else:
                nb_trn_spl += len(v[i])

window_sz = 5
neg_samples = 100
max_session_len = 20
max_nb_his_sess = 0
mini_batch_sz = 200
test_batch_sz = 50
nb_batch = math.ceil(nb_trn_spl / mini_batch_sz)

from models.cross_sess_model import CrossSessRS
crossRS = CrossSessRS(num_items=nb_item, neg_samples=neg_samples, embedding_len=100, ctx_len=2*window_sz,
                      max_sess_len=max_session_len, max_nb_sess=max_nb_his_sess, att_alpha=0.01)

trainModel = crossRS.train_model
trainModel.summary()
trainModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

pred_model = crossRS.predict_model

max_epoch = 10

spl_idx = 0
spl_cnt = 0
for ep in range(max_epoch):
    # Training
    curr_batch = 0
    batch_sz = min(mini_batch_sz, nb_trn_spl)
    his_input = np.zeros([batch_sz, max_nb_his_sess, max_session_len], dtype=np.int32)
    sess_input = np.zeros([batch_sz, 2 * window_sz], dtype=np.int32)
    target_input = np.zeros([batch_sz, neg_samples + 1], dtype=np.int32)
    labels = np.zeros((batch_sz,), dtype=np.int32)

    for k,v in sessionSet.items():
        if len(v) > 1:
            tl = testSet[k]
            for i in range(1, len(v)):
                if i not in tl:
                    sess = v[i]
                    for c in range(len(sess)):
                        target_input[spl_idx, 0] = sess[c]
                        target_input[spl_idx, 1:] = np.random.randint(0, nb_item, neg_samples)
                        ctx_sess = np.concatenate([sess[max(0,c-window_sz):c], sess[(c+1):min(c+1+window_sz,len(sess))]])
                        sess_input[spl_idx, -len(ctx_sess):]  = ctx_sess
                        for j in range(min(i, max_nb_his_sess)):
                            his_sess = v[i-1-j]
                            if len(his_sess) > max_session_len:
                                his_sess = his_sess[:max_session_len]
                            his_input[spl_idx, -(j+1), :len(his_sess)] = his_sess
                        spl_idx += 1
                        if spl_idx == batch_sz:
                            spl_idx = 0
                            curr_batch += 1
                            spl_cnt += batch_sz
                            batch_sz = min(mini_batch_sz, nb_trn_spl - spl_cnt)
                            metrics = trainModel.train_on_batch(x=[sess_input, his_input, target_input], y=labels)

                            if curr_batch % 100 == 0 or curr_batch == nb_batch:
                                print('training %d/%d, acc: %g'%(curr_batch, nb_batch, metrics[1]))

                            his_input = np.zeros([batch_sz, max_nb_his_sess, max_session_len], dtype=np.int32)
                            sess_input = np.zeros([batch_sz, 2 * window_sz], dtype=np.int32)
                            target_input = np.zeros([batch_sz, neg_samples + 1], dtype=np.int32)
                            labels = np.zeros((batch_sz,), dtype=np.int32)
    spl_cnt = 0

    # Evaluation
    curr_batch = 0
    nb_ts_batch = math.ceil(nb_tst_spl_last / test_batch_sz)
    batch_sz = min(test_batch_sz, nb_tst_spl_last)
    his_input = np.zeros([batch_sz, max_nb_his_sess, max_session_len], dtype=np.int32)
    sess_input = np.zeros([batch_sz, 2 * window_sz], dtype=np.int32)
    true_item = np.zeros((batch_sz, ), dtype=np.int32)
    atK = np.arange(5,51,5)
    ranklist = []
    recs= []
    for k, v in testSet.items():
        if v:
            user_sess = sessionSet[k]
            for i in v:
                sess = user_sess[i]
                c = len(sess) - 1
                if c >= 0:
                    true_item[spl_idx] = sess[c]
                    ctx_sess = np.concatenate(
                        [sess[max(0, c - window_sz):c], sess[(c + 1):min(c + 1 + window_sz, len(sess))]])
                    sess_input[spl_idx, -len(ctx_sess):] = ctx_sess
                    for j in range(min(i, max_nb_his_sess)):
                        his_sess = user_sess[i - 1 - j]
                        if len(his_sess) > max_session_len:
                            his_sess = his_sess[:max_session_len]
                        his_input[spl_idx, -(j + 1), :len(his_sess)] = his_sess
                    spl_idx += 1
                    if spl_idx == batch_sz:
                        spl_idx = 0
                        spl_cnt += batch_sz
                        curr_batch += 1
                        scores = pred_model.predict_on_batch(
                            [sess_input, his_input, np.tile(np.arange(0, nb_item + 1), (batch_sz, 1))])
                        for n, score in enumerate(scores):
                            ranklist.append((score > score[true_item[n]]).sum() + 1)
                            recs.append(np.argpartition(score, -10)[-10:])

                        if curr_batch % 10 == 0 or curr_batch == nb_batch:
                            print('Evaluating %d/%d...' % (curr_batch, nb_ts_batch))
                        batch_sz = min(test_batch_sz, nb_tst_spl_last - spl_cnt)
                        his_input = np.zeros([batch_sz, max_nb_his_sess, max_session_len], dtype=np.int32)
                        sess_input = np.zeros([batch_sz, 2 * window_sz], dtype=np.int32)
                        true_item = np.zeros((batch_sz,), dtype=np.int32)
    spl_cnt = 0

    ranklist = np.asarray(ranklist)
    print("AUC: %g" % (np.mean((nb_item - ranklist) / nb_item)))
    print("MRR")
    print(np.mean((1 / ranklist[:, None]) * (ranklist[:, None] <= atK), axis=0))
    print("RECALL")
    print(np.mean(ranklist[:, None] <= atK, axis=0))
    if (ep + 1) % 10 == 0 or (ep + 1) == max_epoch:
        print("Div: %g" % ut.diversity(recs))



curr_batch = 0
nb_ts_batch = math.ceil(nb_tst_spl / test_batch_sz)
batch_sz = min(test_batch_sz, nb_tst_spl)
his_input = np.zeros([batch_sz, max_nb_his_sess, max_session_len], dtype=np.int32)
sess_input = np.zeros([batch_sz, 2 * window_sz], dtype=np.int32)
true_item = np.zeros((batch_sz, ), dtype=np.int32)
atK = np.arange(5,51,5)
ranklist = []
recs= []

for k,v in testSet.items():
    if v:
        user_sess = sessionSet[k]
        for i in v:
            sess = user_sess[i]
            for c in range(len(sess)):
                true_item[spl_idx] = sess[c]
                ctx_sess = np.concatenate(
                    [sess[max(0, c - window_sz):c], sess[(c + 1):min(c + 1 + window_sz, len(sess))]])
                sess_input[spl_idx, -len(ctx_sess):] = ctx_sess
                for j in range(min(i, max_nb_his_sess)):
                    his_sess = user_sess[i - 1 - j]
                    if len(his_sess) > max_session_len:
                        his_sess = his_sess[:max_session_len]
                    his_input[spl_idx, -(j + 1), :len(his_sess)] = his_sess
                spl_idx += 1
                if spl_idx == batch_sz:
                    spl_idx = 0
                    spl_cnt += batch_sz
                    curr_batch += 1
                    scores = pred_model.predict_on_batch(
                        [sess_input, his_input, np.tile(np.arange(0, nb_item + 1), (batch_sz, 1))])
                    for n, score in enumerate(scores):
                        ranklist.append((score > score[true_item[n]]).sum() + 1)
                        recs.append(np.argpartition(score, -10)[-10:])

                    if curr_batch % 10 == 0 or curr_batch == nb_batch:
                        print('Evaluating %d/%d...' % (curr_batch, nb_ts_batch))
                    batch_sz = min(test_batch_sz, nb_tst_spl - spl_cnt)
                    his_input = np.zeros([batch_sz, max_nb_his_sess, max_session_len], dtype=np.int32)
                    sess_input = np.zeros([batch_sz, 2 * window_sz], dtype=np.int32)
                    true_item = np.zeros((batch_sz,), dtype=np.int32)

ranklist = np.asarray(ranklist)
print("AUC: %g" % (np.mean((nb_item - ranklist) / nb_item)))
print("MRR")
print(np.mean((1 / ranklist[:, None]) * (ranklist[:, None] <= atK), axis=0))
print("RECALL")
print(np.mean(ranklist[:, None] <= atK, axis=0))
print("Div: %g" % ut.diversity(recs))