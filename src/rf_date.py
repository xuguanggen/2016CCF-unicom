#! /usr/bin/env python
import sys
import os
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

result_csv_path = 'result/fusai_type_rf_20161210_2.csv'


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
    tr_x ,tr_y = load_data(TRAIN,True)
    df_tr = pd.read_csv(TRAIN,header=0)
    te_x = load_data(TEST,False)
    df_te = pd.read_csv(TEST,header=0)
    df_shop = pd.read_csv(SHOP_PROFILE,header=0)

    dic_tr_x = {}
    dic_tr_y = {}
    for i in range(tr_x.shape[0]):
        cur_shopid = int(df_tr['SHOPID'][i])
        if cur_shopid ==0 :
            continue
        cur_date = int(df_tr['DATE'][i])
        if cur_date in dic_tr_x.keys():
            sub_tr_x = dic_tr_x[cur_date]
            sub_tr_y = dic_tr_y[cur_date]
            sub_tr_x.append(tr_x[i])
            sub_tr_y.append(tr_y[i])
            dic_tr_x[cur_date] = sub_tr_x
            dic_tr_y[cur_date] = sub_tr_y
        else:
            sub_tr_x = []
            sub_tr_y = []
            sub_tr_x.append(tr_x[i])
            sub_tr_y.append(tr_y[i])
            dic_tr_x[cur_date] = sub_tr_x
            dic_tr_y[cur_date] = sub_tr_y

    dic_rf = {}
    for cur_date ,sub_tr_x in dic_tr_x.items():
        sub_tr_x = np.array(sub_tr_x)
        sub_tr_y = np.array(dic_tr_y[cur_date])
        rf = RandomForestClassifier(
                n_estimators = 200,
                max_depth = 11,
                min_samples_split =2,
                bootstrap =True,
                warm_start = True,
                max_features = 'sqrt',
                criterion='entropy',
                class_weight = 'balanced',
                n_jobs = -1
                ).fit(sub_tr_x,sub_tr_y)
        dic_rf[cur_date] = rf



    te_pred_list = []
    for i in range(te_x.shape[0]):
        duration = df_te['DURATION'][i]
        cur_date = df_te['DATE'][i]
        if duration <= 15:
            te_pred_list.append('')
        else:
            this_pred_dic = {}
            for cur_date,sub_tr_x in dic_tr_x.items():
                rf = dic_rf[cur_date]
                this_te_pred = (rf.predict(te_x[i]))
                pred_shopid = int(this_te_pred[0])
                if pred_shopid not in this_pred_dic.keys():
                    this_pred_dic[pred_shopid] = 1
                else:
                    this_pred_dic[pred_shopid] += 1
            sorted_this_pred_dic = sorted(this_pred_dic.iteritems(),key=lambda x:x[1],reverse=True)

            most_shopid = sorted_this_pred_dic[0][0]
            te_pred_list.append(most_shopid)
        print('te_idx:'+str(i))

    save_results(result_csv_path,te_pred_list)



if __name__=='__main__':
    start = time()
    run()
    end = time()
    print('Time:\t'+str((end-start)/3600)+' Hours')
