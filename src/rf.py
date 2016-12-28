#! /usr/bin/env python

from time import time
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

from utils import load_data,save_results
from config import TRAIN,TEST
from config import Features


Model_Name = 'fusai_rf_20161202_6'
result_csv_path = 'result/'+Model_Name+'.csv'




def run():
    tr_x ,tr_y = load_data(TRAIN,True)
    te_x = load_data(TEST,False)
    rf = RandomForestClassifier(
            n_estimators = 500,
            max_depth = 11,
            min_samples_split =2,
            bootstrap =True,
            warm_start = True,
            max_features = 'sqrt',
            criterion='entropy',
            class_weight = 'balanced',
            n_jobs = -1
            )
    #rf.fit(tr_x,tr_y)
    ##feature_importances = rf.feature_importances_
    ##dic_feature_importances = dict(zip(Features,feature_importances))
    ##dic = sorted(dic_feature_importances.iteritems(),key=lambda d:d[1],reverse=True)
    ##print('===========================\n')
    ##print('feature_importances:')
    ##for i in range(len(dic)):
    ##    print(dic[i][0]+":\t"+str(dic[i][1]))
    #te_pred = rf.predict(te_x)
    #save_results(result_csv_path,te_pred)

    #sum_acc = 0
    #cv = 10
    #kf = KFold(tr_x.shape[0],n_folds = cv,shuffle=True)
    #for train,val in kf:
    #    x_tr,x_val,y_tr,y_val = tr_x[train],tr_x[val],tr_y[train],tr_y[val]
    #    rf.fit(x_tr,y_tr)
    #    pred_val = rf.predict(x_val)
    #    true_count = 0
    #    for i in range(len(y_val)):
    #        if y_val[i] == pred_val[i]:
    #            true_count += 1
    #    acc = true_count*1.0/len(pred_val)
    #    sum_acc += acc
    #    print('acc :'+ str(acc))
    #print('avg acc:'+str(sum_acc/cv))
    cv = 10
    scores = cross_val_score(rf,tr_x,tr_y,cv=cv,scoring='f1_weighted')
    avg_score = sum(scores)/cv
    print(str(scores))
    print('scores:\t'+str(avg_score))
    #while True:
    #    #rf.fit(tr_x,tr_y)
    #    scores = cross_val_score(rf,tr_x,tr_y,cv=cv,scoring='f1_weighted')
    #    avg_score = sum(scores)/cv
    #    print(str(scores))
    #    print('scores:\t'+str(avg_score))
    #    if avg_score > 0.6:
    #        te_pred = rf.predict(te_x)
    #        save_results(result_csv_path,te_pred)
    #        break

    #print(str(scores))
    #print(str(sum(scores)/cv))
    ########################################################################################


if __name__=='__main__':
    start = time()
    run()
    end = time()
    print('Time:\t'+str((end-start)/3600)+' Hours')
