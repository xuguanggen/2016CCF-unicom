#! /usr/bin/env python
import sys
import pandas as pd
import numpy as np
sys.path.append("..")
from config import *
NUM_CANDIDATE = 8

def load_shop():
    shop_dict = {}
    f = open('shopid_key.txt','r')
    for line in f:
        data = line.strip().split(":")
        shop_dict[int(data[1])] = int(data[0])
    return shop_dict


def MatchShopIdForPredictTxt():
    shop_dict = load_shop()

    f_pred = open('pred.txt','r')
    f_te = open('TEST_NEW_SVM.txt','r')
    f_out = open('true_pred.txt','w')
    
    pred_list = f_pred.readlines()
    test_list = f_te.readlines()

    for cur_predline,cur_testline in zip(pred_list,test_list):
        cur_proba = float(cur_predline.strip())
        data = cur_testline.strip().split(" ")
        shopid = int(data[-1].split(":")[0])
        true_shopid = shop_dict[shopid]
        f_out.write(str(true_shopid)+","+str(cur_proba)+"\n")

def recommend_shop():
    f = open('true_pred.txt','r')
    recommend_list = []
    proba_list = f.readlines()
    i = 0
    while i < len(proba_list):
        dic = {}
        for j in range(NUM_CANDIDATE):
            cur_shop = proba_list[i+j].strip().split(",")[0]
            cur_prob = float(proba_list[i+j].strip().split(",")[1])
            if cur_shop not in dic.keys():
                dic[cur_shop] = cur_prob
            else:
                dic[cur_shop] += cur_prob
        dic_sorted = sorted(dic.iteritems(),key=lambda d:d[1],reverse=True)
        if dic_sorted[0][0] == '0':
            recommend_list.append('')
        else:
            recommend_list.append(dic_sorted[0][0])
        i += NUM_CANDIDATE
    df_recommend = pd.DataFrame()
    df_test = pd.read_csv(TEST,header=0)
    df_recommend['USERID'] = df_test['USERID']
    df_recommend['SHOPID'] = recommend_list
    df_recommend['ARRIVAL_TIME'] = df_test['ARRIVAL_TIME']
    df_recommend.to_csv('recommond.csv',index=False)



def run():
    MatchShopIdForPredictTxt()
    recommend_shop()

if __name__=='__main__':
    run()

