#! /usr/bin/env python
from time import time
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np


def run():
    tr_data = np.loadtxt('../new/TRAIN_LRFORMAT.txt')
    te_data = np.loadtxt('../new/TEST_LRFORMAT.txt')

    tr_x = tr_data[:,1:]
    tr_y = tr_data[:,0]
    te_x = te_data[:,1:]

    lr = LogisticRegression(
            solver='liblinear',
            multi_class='ovr',
            class_weight='balanced',
            penalty='l2',
            n_jobs=-1)
    #te_pred = lr.predict_proba(te_x)
    cv = 10
    scores = cross_val_score(lr,tr_x,tr_y,cv=cv,scoring='accuracy')
    print(str(scores))
    #np.savetxt('result/te_lr.txt',te_pred)


if __name__=='__main__':
    start = time()
    run()
    end = time()
    print('time :\t'+str((end-start)/3600)+' hours')
