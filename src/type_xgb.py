#! /usr/bin/env python

from time import time
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

from utils import load_data,save_results
from config import TRAIN,TEST,SHOP_PROFILE
from config import Features

result_csv_path = 'result/type_rf_20161128_2.csv'


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


def run():
    tr_x ,shop_y = load_data(TRAIN,True)
    df_tr = pd.read_csv(TRAIN,header=0)
    tr_y = np.array(df_tr['SHOP_CLASS'].values)
    te_x = load_data(TEST,False)
    df_te = pd.read_csv(TEST,header=0)
    df_shop = pd.read_csv(SHOP_PROFILE,header=0)
    df_tr_candidate = pd.read_csv('../new/TRAIN_NEIGHUSER_SHOPID.csv',header=0)
    df_te_candidate = pd.read_csv('../new/TEST_NEIGHUSER_SHOPID.csv',header=0)
    
    random_seed = 1221 
    params ={
        'booster': 'gbtree',
        'eta': 0.02, # ?
        'gamma': 0.1,
        'max_depth': 7,
        'min_child_weight': 3, # ?
        'subsample': 0.7,
        'colsample_bytree': 0.4,
        'lambda': 550,
        'objective': 'multi:softmax',
        'seed': random_seed,
        'eval_metrix':'mlogclass',
        'early_stopping_rounds': 100,
        'num_class':11
    }
    
    sum_acc = 0
    cv = 10
    kf = KFold(tr_x.shape[0],n_folds = cv)
    for train,val in kf:
        x_tr,x_val,y_tr,y_val = tr_x[train],tr_x[val],tr_y[train],tr_y[val]
        
        dtrain = xgb.DMatrix(x_tr,y_tr)
        dtest = xgb.DMatrix(x_val)
        watchlist = [(dtrain,'train')]
        bst = xgb.train(params,dtrain,num_boost_round=1000,evals=watchlist)
        pred_val = bst.predict(dtest)
        true_count = 0
        for i,idx in zip(range(len(y_val)),val):
            print(str(pred_val[i]))
            if pred_val[i]==df_tr['SHOP_CLASS'][idx]:
                true_count += 1

        #for i,idx in zip(range(len(y_val)),val):
        #    pred_shoptype = pred_val[i]
        #    true_shopid = df_tr['SHOPID'][idx]
        #    if pred_shoptype==0 and true_shopid==0:
        #        true_count += 1
        #        continue
        #    #neighuser_shopid = list(eval(df_tr['NEIGHUSER_SHOPID'][idx]))
        #    neighuser_shopid = getCandidateShopId(df_tr_candidate,idx)
        #    pred_shopid = neighuser_shopid[0]
        #    for j in range(len(neighuser_shopid)):
        #        cur_shopid = neighuser_shopid[j]
        #        if cur_shopid ==0:
        #            if pred_shoptype ==0:
        #                pred_shopid = 0
        #            continue
        #        cur_shoptype = (df_shop[df_shop['ID']==cur_shopid]['CLASSIFICATION'].values)[0]
        #        if cur_shoptype == pred_shoptype:
        #            pred_shopid == cur_shopid
        #            break
        #    if pred_shopid==true_shopid:
        #        true_count += 1


        acc = true_count*1.0/len(pred_val)
        sum_acc += acc
        print('acc :'+ str(acc))
    print('avg acc:'+str(sum_acc/cv))
   



if __name__=='__main__':
    start = time()
    run()
    end = time()
    print('Time:\t'+str((end-start)/3600)+' Hours')
