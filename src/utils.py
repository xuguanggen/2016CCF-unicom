#! /usr/bin/env python

import datetime
from time import time
import time

import numpy as np
import math as Math
import pandas as pd

from config import List_Features
from config import Common_Features
from config import TRAIN,TEST,SHOP_PROFILE
from config import MYTEST


def get_hourAndweek(date):
    date = time.strptime(date,"%Y%m%d%H%M%S")
    week = int(datetime.datetime(date[0],date[1],date[2]).strftime('%w'))
    date = datetime.datetime(date[0],date[1],date[2],date[3],date[4],date[5])
    hour = date.hour
    return hour,week

def IsAwayShopCenter(cur_lon,cur_lat,center_lon,center_lat,dest_lon,dest_lat):
    destance1,dir1 = caldistance(cur_lon,cur_lat,center_lon,center_lat)
    destance2,dir2 = caldistance(dest_lon,dest_lat,center_lon,center_lat)

    if destance1 <= destance2:
        return 1
    return 0




def rad(d):
    return d * Math.pi /180.0

def caldistance(lon1,lat1,lon2,lat2):
    lon1 = float(lon1)
    lat1 = float(lat1)
    lon2 = float(lon2)
    lat2 = float(lat2)
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lon1) - rad(lon2)
    s = 2 * Math.asin(Math.sqrt(Math.pow(Math.sin(a/2),2) +
        Math.cos(radLat1)*Math.cos(radLat2)*Math.pow(Math.sin(b/2),2)))
    s = s*6378.137
    s = round(s*10000) /10000


    #### calc direction ######
    direction = 0
    if lon2 > lon1 and lat2 >= lat1:
        direction = 1
    elif lon2 > lon1 and lat2 < lat1:
        direction = 2
    elif lon2 <= lon1 and lat2 < lat1:
        direction = 3
    else:
        direction = 4


    return s ,direction


def caltime(date1,date2):
    date1 = time.strptime(date1,"%Y%m%d%H%M%S")
    date2 = time.strptime(date2,"%Y%m%d%H%M%S")
    date1 = datetime.datetime(date1[0],date1[1],date1[2],date1[3],date1[4],date1[5])
    date2 = datetime.datetime(date2[0],date2[1],date2[2],date2[3],date2[4],date2[5])
    return abs((date2 - date1).total_seconds())








def get_list_features(csv_path):
    df = pd.read_csv(csv_path,header=0)
    feature_neighshop_id = []
    for i in range(len(df)):
        cur_id = list(eval(df['NEIGHSHOP_ID'][i]))
        feature_neighshop_id.append(cur_id)
    feature_neighshop_id = np.array(feature_neighshop_id)
    feature_list = np.hstack([feature_neighshop_id])
    for i in range(1,len(List_Features)):
        cur_feature_list = []
        for j in range(len(df)):
            cur_feature = list(eval(df[List_Features[i]][j]))
            cur_feature_list.append(cur_feature)
        cur_feature_list = np.array(cur_feature_list)
        feature_list = np.hstack([feature_list,cur_feature_list])
    return feature_list


def get_common_features(csv_path):
    df = pd.read_csv(csv_path,header=0)
    feature_common = []
    for i in range(len(Common_Features)):
        feature_name = Common_Features[i]
        cur_feature = np.array(df[feature_name]).reshape(len(df),1)
        feature_common.append(cur_feature)
    feature_common = np.hstack(feature_common)
    return feature_common


def get_tr_y(csv_path):
    df = pd.read_csv(csv_path,header=0)
    return np.array(df['SHOPID'])


def load_data(csv_path,isTrain):
    feature_list = get_list_features(csv_path)
    feature_common = get_common_features(csv_path)
    if isTrain:
        tr_y = get_tr_y(csv_path)
        #tr_hyk = np.loadtxt('../new/newtrfeature.txt')
        tr_x = np.hstack([feature_list,feature_common])
        return tr_x,tr_y
    else:
        #te_hyk = np.loadtxt('../new/newtefeature.txt')
        te_x = np.hstack([feature_list,feature_common])
        return te_x


def save_results(result_csv_path,pred_y):
    df = pd.read_csv(TEST,header=0)
    df_result = pd.DataFrame()
    df_result['USERID'] = df['USERID']
    df_result['SHOPID'] =  pred_y
    df_result['ARRIVAL_TIME'] = df['ARRIVAL_TIME']
    df_result.to_csv(result_csv_path,index=False)


def similarity(colname,n):
    df_train = pd.read_csv(TRAIN,header=0)
    df_shop = pd.read_csv(SHOP_PROFILE,header=0)
    x = np.zeros((n,10))
    num_null_ofthisAttr = [0]*n
    
    for i in range(len(df_train)):
        att = int(df_train[colname][i])-1
        shopid = (df_train['SHOPID'][i])
        if shopid == 0:
            continue
        shopclass = int([df_shop[df_shop['ID']==shopid]['CLASSIFICATION'].values][0])-1
        x[att][shopclass] += 1
    num_count = [0]*n
    for i in range(n):
        for j in range(10):
            if x[i][j]>0:
                num_count[i] += 1

    for i in range(n):
        y = sum(x[i])
        for j in range(10):
            x[i][j] =(x[i][j]+1)/(y+num_count[i])
    for i in range(n):
        s = "["
        for j in range(10):
            s += str(x[i][j])+','
        print(s[:-1]+']')



def generateMyTestFile():
    df_te = pd.read_csv(TEST,header=0)
    uid_list = []
    shopid_list = []
    time_list = []
    for i in range(len(df_te)):
        cur_uid = df_te['USERID'][i]
        cur_time = df_te['ARRIVAL_TIME'][i]
        neighuser_shoplist = list(eval(df_te['NEIGHUSER_SHOPID'][i]))
        for j in range(len(neighuser_shoplist)):
            uid_list.append(cur_uid)
            shopid_list.append(neighuser_shoplist[j])
            time_list.append(cur_time)
    df_myte = pd.DataFrame()
    df_myte['USERID'] = uid_list
    df_myte['SHOPID'] = shopid_list
    df_myte['ARRIVAL_TIME'] = time_list
    df_myte.to_csv(MYTEST,index=False)




if __name__=='__main__':
    #load_data('../data/TRAIN.csv',True)
    #load_data('../data/TEST2.csv',False)
    #get_weekday_time('20161016001030')
    similarity('INCOME',4)
    #generateMyTestFile()
