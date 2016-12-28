#! /usr/bin/env python

from time import time
import numpy as np
import pandas as pd


from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

from utils import load_data,save_results
from config import TRAIN,TEST,SHOP_PROFILE
from config import Features

result_csv_path = 'result/type_rf_20161215_7.csv'


def getCandidateShopId(df,Idx):
    candidate_shopid_list = []
    candidate_neighuser = list(eval(df['NEIGHUSER_SHOPID'][Idx]))
    candidate_neighshop = list(eval(df['NEIGHSHOP_ID'][Idx]))

    dic_shopid = {}
    for i in range(len(candidate_neighuser)):
        if candidate_neighuser[i] != 0:
            candidate_shopid_list.append(candidate_neighuser[i])
            dic_shopid[candidate_neighuser[i]] = 1

    for i in range(len(candidate_neighshop)):
        cur_shopid = candidate_neighshop[i]
        if cur_shopid not in dic_shopid.keys():
            candidate_shopid_list.append(cur_shopid)
    return candidate_shopid_list


def predict_type():
    tr_x ,shop_y = load_data(TRAIN,True)
    df_tr = pd.read_csv(TRAIN,header=0)
    tr_y = np.array(df_tr['SHOP_CLASS'].values)
    te_x = load_data(TEST,False)
    df_te = pd.read_csv(TEST,header=0)
    df_shop = pd.read_csv(SHOP_PROFILE,header=0)
    df_tr_candidate = pd.read_csv('../new/TRAIN_NEIGHUSER_SHOPID.csv',header=0)
    df_te_candidate = pd.read_csv('../new/TEST_NEIGHUSER_SHOPID.csv',header=0)
    
    rf = RandomForestClassifier(
            n_estimators = 215,
            max_depth = 11,
            min_samples_split =2,
            bootstrap =True,
            warm_start = True,
            max_features = 'sqrt',
            criterion='entropy',
            class_weight = 'balanced',
            n_jobs = -1
            )

    tr_one_x = []
    tr_one_y = []
    for i in range(tr_x.shape[0]):
        if df_tr['SHOPID'][i] != 0:
            tr_one_x.append(tr_x[i])
            tr_one_y.append(tr_y[i])
    tr_one_x = np.array(tr_one_x)
    tr_one_y = np.array(tr_one_y)

    rf.fit(tr_one_x,tr_one_y)
    te_pred_shoptype = rf.predict(te_x)
    for i in range(te_x.shape[0]):
        cur_duration = df_te['DURATION'][i]
        if cur_duration <= 15:
            te_pred_shoptype[i] = 0
    return te_pred_shoptype

    #count = 0
    #for i in range(te_pred_shoptype.shape[0]):
    #    if te_pred_shoptype[i] ==0:
    #        count += 1
    #print(str(count))
    #print('te_pred_shoptype lenght:'+str(te_pred_shoptype.shape[0]))
    #te_recommend_shopid_list = []
    #for i in range(te_pred_shoptype.shape[0]):
    #    cur_pred_shoptype = te_pred_shoptype[i]
    #    if cur_pred_shoptype == 0:
    #        print(str(i)+':0')
    #        te_recommend_shopid_list.append('')
    #    else:
    #        candidate_shopid = getCandidateShopId(df_te_candidate,i)
    #        j = 0
    #        while j <len(candidate_shopid):
    #            cur_shopid = (candidate_shopid[j])
    #            cur_shoptype = (df_shop[df_shop['ID']==cur_shopid]['CLASSIFICATION'].values)[0]
    #            if cur_shoptype == cur_pred_shoptype:
    #                print(str(i)+'='+str(cur_shopid))
    #                te_recommend_shopid_list.append(cur_shopid)
    #                break
    #            j = j+1
    #        #print(str(i)+"****"+str(j))
    #        if j == len(candidate_shopid):
    #            print(str(i)+'-'+str(candidate_shopid[0]))
    #            te_recommend_shopid_list.append(candidate_shopid[0])
    #print(str(len(te_recommend_shopid_list)))
    #save_results(result_csv_path,te_recommend_shopid_list)

    sum_acc = 0
    cv = 10
    kf = KFold(tr_x.shape[0],n_folds = cv,shuffle=True)
    for train,val in kf:
        x_tr,x_val,y_tr,y_val = tr_x[train],tr_x[val],tr_y[train],tr_y[val]
        rf.fit(x_tr,y_tr)


        pred_val = rf.predict(x_val)
        true_count = 0
        #for i,idx in zip(range(len(y_val)),val):
        #    #print(str(pred_val[i]))
        #    if pred_val[i]==df_tr['SHOP_CLASS'][idx]:
        #        true_count += 1

        for i,idx in zip(range(len(y_val)),val):
            pred_shoptype = pred_val[i]
            true_shopid = df_tr['SHOPID'][idx]
            if pred_shoptype==0 and true_shopid==0:
                true_count += 1
                continue
            #neighuser_shopid = list(eval(df_tr['NEIGHUSER_SHOPID'][idx]))
            neighuser_shopid = getCandidateShopId(df_tr_candidate,idx)
            pred_shopid = neighuser_shopid[0]
            for j in range(len(neighuser_shopid)):
                cur_shopid = neighuser_shopid[j]
                if cur_shopid ==0:
                    if pred_shoptype ==0:
                        pred_shopid = 0
                    continue
                cur_shoptype = (df_shop[df_shop['ID']==cur_shopid]['CLASSIFICATION'].values)[0]
                if cur_shoptype == pred_shoptype:
                    pred_shopid == cur_shopid
                    break
            if pred_shopid==true_shopid:
                true_count += 1


        acc = true_count*1.0/len(pred_val)
        sum_acc += acc
        print('acc :'+ str(acc))
    print('avg acc:'+str(sum_acc/cv))



if __name__=='__main__':
    start = time()
    predict_type()
    end = time()
    print('Time:\t'+str((end-start)/3600)+' Hours')
