#! /usr/bin/env python
import os
from time import time
import numpy as np
import pandas as pd

import cPickle as pickle
from sklearn.cluster import KMeans,MeanShift


from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score


from xgboost import XGBClassifier


from utils import load_data,save_results
from config import TRAIN,TEST,SHOP_PROFILE
from config import Features



tr_x,tr_y = load_data(TRAIN,True)
te_x = load_data(TEST,False)

df_tr = pd.read_csv(TRAIN,header=0)
df_te = pd.read_csv(TEST,header=0)



def cluster(tr_lonlat_list,num_clusters,thiskey):

    dict_fileName = r'pkl/dict_'+thiskey+".pkl"
    tr_lonlat_list = np.array(tr_lonlat_list)
    kmeans = KMeans(n_clusters=num_clusters,n_jobs=-1).fit(tr_lonlat_list)
    #mf = MeanShift(bandwidth=0.001,bin_seeding=True,min_bin_freq=5).fit(tr_lonlat_list)

    lonlat_cluster_dict = {}
    for i in range(tr_lonlat_list.shape[0]):
        key = str(tr_lonlat_list[i][0])+":"+str(tr_lonlat_list[i][1])
        lonlat_cluster_dict[key] = kmeans.labels_[i]
    f_w = open(dict_fileName,'w')
    pickle.dump(lonlat_cluster_dict,f_w)
    return lonlat_cluster_dict


def run_cluster_model(thiskey,this_lon,this_lat,sample_te_x,num_clusters):

    dict_fileName = r'pkl/dict_'+thiskey+".pkl"
    this_tr_x_fileName = r'pkl/x_'+thiskey+".pkl"
    this_tr_y_fileName = r'pkl/y_'+thiskey+'.pkl'
    this_tr_lonlat_fileName = r'pkl/tr_lonlat_list_'+thiskey+".pkl"

    lonlat_cluster_dict = {}
    this_tr_x = []
    this_tr_y = []
    tr_lonlat_list = []

    if os.path.exists(dict_fileName):
        f_dict = open(dict_fileName,'r')
        f_x = open(this_tr_x_fileName,'r')
        f_y = open(this_tr_y_fileName,'r')
        f_tr_lonlat_list = open(this_tr_lonlat_fileName,'r')

        lonlat_cluster_dict = pickle.load(f_dict)
        this_tr_x = pickle.load(f_x)
        this_tr_y = pickle.load(f_y)
        tr_lonlat_list = pickle.load(f_tr_lonlat_list)
    else:
        for i in range(tr_x.shape[0]):
            if df_tr['SHOPID'][i] == 0:
                continue
            income = str(df_tr['INCOME'][i])
            enter = str(df_tr['ENTERTAINMENT'][i])
            baby = str(df_tr['BABY'][i])
            shopping = str(df_tr['SHOPPING'][i])
            key = income+" "+enter+" "+baby+" "+shopping
            if key == thiskey:
                lon = df_tr['LON'][i]
                lat = df_tr['LAT'][i]
                tr_lonlat_list.append([lon,lat])
                this_tr_x.append(tr_x[i])
                this_tr_y.append(tr_y[i])
        
        #### add testset lonlat #########

        for i in range(te_x.shape[0]):
            income = str(df_te['INCOME'][i])
            enter = str(df_te['ENTERTAINMENT'][i])
            baby = str(df_te['BABY'][i])
            shopping = str(df_te['SHOPPING'][i])
            key = income+" "+enter+" "+baby+" "+shopping
            if key == thiskey:
                lon = df_te['LON'][i]
                lat = df_te['LAT'][i]
                tr_lonlat_list.append([lon,lat])

        lonlat_cluster_dict = cluster(tr_lonlat_list,num_clusters,thiskey)
        f_x = open(this_tr_x_fileName,'w')
        f_y = open(this_tr_y_fileName,'w')
        f_lonlat_list = open(this_tr_lonlat_fileName,'w')
        pickle.dump(this_tr_x,f_x)
        pickle.dump(this_tr_y,f_y)
        pickle.dump(tr_lonlat_list,f_lonlat_list)





    given_cluster_label = lonlat_cluster_dict[str(this_lon)+":"+str(this_lat)]

    sub_this_tr_x = []
    sub_this_tr_y = []
    for i in range(len(this_tr_x)):
        lon = tr_lonlat_list[i][0]
        lat = tr_lonlat_list[i][1]
        key = str(lon)+":"+str(lat)
        cluster_label = lonlat_cluster_dict[key]
        if given_cluster_label == cluster_label:
            sub_this_tr_x.append(this_tr_x[i])
            sub_this_tr_y.append(this_tr_y[i])

    sub_this_tr_x = np.array(sub_this_tr_x)
    sub_this_tr_y = np.array(sub_this_tr_y)

    rf = RandomForestClassifier(
            n_estimators = 165,
            max_depth = 12,
            min_samples_split =2,
            bootstrap =True,
            warm_start = True,
            max_features = 'sqrt',
            criterion='entropy',
            class_weight = 'balanced',
            n_jobs = -1
            ).fit(sub_this_tr_x,sub_this_tr_y)
   # xgb = XGBClassifier(
   #         max_depth = 12,
   #         learning_rate = 0.05,
   #         n_estimators = 160,
   #         silent = False,
   #         objective = 'multi:softmax',
   #         nthread = -1,
   #         gamma = 0,
   #         min_child_weight = 1,
   #         max_delta_step = 0.7,
   #         subsample = 1,
   #         colsample_bytree=1,
   #         reg_lambda=1,
   #         base_score=0.1,
   #         scale_pos_weight=1,
   #         seed = 1227
   #         ).fit(sub_this_tr_x,sub_this_tr_y)
    this_predict_shopid = rf.predict(sample_te_x)
    return this_predict_shopid[0]



def run_simple_model(thiskey,sample_te_x):
    rf_fileName = 'pkl/RF_sample_'+thiskey+'.pkl'
    if os.path.exists(rf_fileName):
        rf = joblib.load(rf_fileName)
        this_predict_shopid = rf.predict(sample_te_x)
        return this_predict_shopid[0]
    else:
        sub_this_tr_x = []
        sub_this_tr_y = []

        for i in range(tr_x.shape[0]):
            if df_tr['SHOPID'][i] == 0:
                continue
            income = str(df_tr['INCOME'][i])
            enter = str(df_tr['ENTERTAINMENT'][i])
            baby = str(df_tr['BABY'][i])
            shopping = str(df_tr['SHOPPING'][i])
            key = income+" "+enter+" "+baby+" "+shopping
            if key == thiskey:
                sub_this_tr_x.append(tr_x[i])
                sub_this_tr_y.append(tr_y[i])

        sub_this_tr_x = np.array(sub_this_tr_x)
        sub_this_tr_y = np.array(sub_this_tr_y)
        rf = RandomForestClassifier(
                n_estimators = 12,
                min_samples_split =2,
                bootstrap =True,
                warm_start = True,
                max_features = 'sqrt',
                criterion='entropy',
                class_weight = 'balanced',
                n_jobs = -1
                ).fit(sub_this_tr_x,sub_this_tr_y)
        joblib.dump(rf,rf_fileName)
        this_predict_shopid = rf.predict(sample_te_x)
        return this_predict_shopid[0]
    return 999999







#if __name__=='__main__':
#    start = time()
#    run("1 2 1 5")
#    end = time()
#    print('Time:\t'+str((end-start)/3600)+' Hours')
