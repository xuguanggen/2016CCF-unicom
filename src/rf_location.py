#! /usr/bin/env python

from time import time
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans,MeanShift

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

from utils import load_data,save_results
from config import TRAIN,TEST,SHOP_PROFILE
from config import Features



result_csv_path = 'result/location_rf_20161204_9.csv'
NUM_CLUSTERS = 25



def cluster(lonlat_list):


    #dic_lonlat = {}
    #for i in range(len(tr_lonlat_list)):
    #    lon = tr_lonlat_list[i][0]
    #    lat = tr_lonlat_list[i][1]
    #    key = str(lon)+":"+str(lat)
    #    if key not in dic_lonlat.keys():
    #        lonlat_list.append([lon,lat])
    #        dic_lonlat[key] = 1

    #for i in range(len(te_lonlat_list)):
    #    lon = te_lonlat_list[i][0]
    #    lat = te_lonlat_list[i][1]
    #    key = str(lon)+":"+str(lat)
    #    if key not in dic_lonlat.keys():
    #        lonlat_list.append([lon,lat])
    #        dic_lonlat[key] = 1
    
    #lonlat_list = np.array(lonlat_list)

    #kmeans = KMeans(n_clusters=NUM_CLUSTERS,n_jobs=-1).fit(lonlat_list)
    mf = MeanShift().fit(lonlat_list)

    lonlat_cluster_dict = {}
    for i in range(lonlat_list.shape[0]):
        key = str(lonlat_list[i][0])+":"+str(lonlat_list[i][1])
        lonlat_cluster_dict[key] = mf.labels_[i]


    #for i in range(NUM_CLUSTERS):
    #    count = 0
    #    for k,v in lonlat_cluster_dict.items():
    #        if i == v:
    #            count += 1
    #    print('cluster:'+str(i)+'\tcount:'+str(count))
    return lonlat_cluster_dict


def run():
    tr_x,tr_y = load_data(TRAIN,True)
    te_x = load_data(TEST,False)

    df_tr = pd.read_csv(TRAIN,header=0)
    df_te = pd.read_csv(TEST,header=0)

    tr_lonlat_list = np.array(df_tr[['LON','LAT']].values)
    te_lonlat_list = np.array(df_te[['LON','LAT']].values)
    lonlat_list = np.vstack([tr_lonlat_list,te_lonlat_list])
    print(str(tr_lonlat_list.shape))
    print(str(te_lonlat_list.shape))
    print(str(lonlat_list.shape))

    lonlat_cluster_dict = cluster(lonlat_list)

    dict_tr_x = {}
    dict_tr_y = {}
    for cluster_label in range(NUM_CLUSTERS):
        sub_tr_x = []
        sub_tr_y = []
        for i in range(tr_x.shape[0]):
            lon = df_tr['LON'][i]
            lat = df_tr['LAT'][i]
            key = str(lon)+":"+str(lat)
            c_label = lonlat_cluster_dict[key]
            if c_label == cluster_label:
                sub_tr_x.append(tr_x[i])
                sub_tr_y.append(tr_y[i])
        sub_tr_x = np.array(sub_tr_x)
        sub_tr_y = np.array(sub_tr_y)
        dict_tr_x[cluster_label] = sub_tr_x
        dict_tr_y[cluster_label] = sub_tr_y
        #print('cluster:'+str(cluster_label)+'\tcount:'+str(len(sub_tr_x)))

    rf = RandomForestClassifier(
            n_estimators = 150,
            max_depth = 11,
            min_samples_split =2,
            bootstrap =True,
            warm_start = True,
            max_features = 'sqrt',
            criterion='entropy',
            class_weight = 'balanced',
            n_jobs = -1
            )
    dict_rf = {}
    for cluster_label,sub_tr_x in dict_tr_x.items():
        sub_tr_x = np.array(sub_tr_x)
        sub_tr_y = np.array(dict_tr_y[cluster_label])
        rf = rf.fit(sub_tr_x,sub_tr_y)
        dict_rf[cluster_label] = rf
        #cv = 10
        #scores = cross_val_score(rf,sub_tr_x,sub_tr_y,cv=cv,scoring='f1_weighted')
        #avg_score = sum(scores)/cv
        #avg_score_list.append(avg_score*(1.0*sub_tr_x.shape[0])/(tr_x.shape[0]))
        #print(str(cluster_label)+":"+str(sub_tr_x.shape[0]))
        #print(str(scores))
        #print('scores:\t'+str(avg_score)+'\n\n\n')

    #print('max score:\t'+str(max(avg_score_list)))
    #print('min score:\t'+str(min(avg_score_list)))
    #print('total avg:\t'+str(sum(avg_score_list)/NUM_CLUSTERS))

    te_pred = []
    for i in range(te_x.shape[0]):
        lon = df_te['LON'][i]
        lat = df_te['LAT'][i]
        key = str(lon)+":"+str(lat)
        c_label = lonlat_cluster_dict[key]
        cur_rf = dict_rf[cluster_label]
        cur_pred = cur_rf.predict(te_x[i])
        te_pred.append(cur_pred[0])
    save_results(result_csv_path,te_pred)


if __name__=='__main__':
    start = time()
    run()
    end = time()
    print('Time:\t'+str((end-start)/3600)+' Hours')
