#! /usr/bin/env python

from time import time
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans


from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

from utils import load_data,save_results
from config import TRAIN,TEST,SHOP_PROFILE
from config import Features



result_csv_path = 'result/fusai_userWithlocation_rf_20161204_8.csv'
NUM_CLUSTERS = 5


def findMostSimilaritySameUserAttr(dict_tr_x,attr_key):
    given_cluster_label = attr_key.split(' ')[0]
    given_income = attr_key.split(' ')[1]
    given_entertainment = attr_key.split(' ')[2]
    given_baby = attr_key.split(' ')[3]
    given_shopping = attr_key.split(' ')[4]
    
    similarity_score = {}
    max_attr_num = 0
    max_attr_count = 0
    most_similarity_key = ""
    for key,sub_tr_x in dict_tr_x.items():
        cur_cluster_label = key.split(' ')[0]
        cur_income = key.split(' ')[1]
        cur_entertainment = key.split(' ')[2]
        cur_baby = key.split(' ')[3]
        cur_shopping = key.split(' ')[4]
        if cur_cluster_label != given_cluster_label:
            continue
        same_attr_num = 0
        if cur_income == given_income:
            same_attr_num += 1
        if cur_entertainment == given_entertainment:
            same_attr_num += 1
        if cur_baby == given_baby:
            same_attr_num += 1
        if cur_shopping == given_shopping:
            same_attr_num += 1

        if same_attr_num > max_attr_num:
            max_attr_num = same_attr_num
            max_attr_count = len(sub_tr_x)
            most_similarity_key = key
        
        if same_attr_num == max_attr_num:
            if len(sub_tr_x) > max_attr_count:
                max_attr_num = same_attr_num
                max_attr_count = len(sub_tr_x)
                most_similarity_key = key
    return most_similarity_key



def cluster(lonlat_list):

    kmeans = KMeans(n_clusters=NUM_CLUSTERS,n_jobs=-1).fit(lonlat_list)

    lonlat_cluster_dict = {}
    for i in range(lonlat_list.shape[0]):
        key = str(lonlat_list[i][0])+":"+str(lonlat_list[i][1])
        lonlat_cluster_dict[key] = kmeans.labels_[i]


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
        for i in range(tr_x.shape[0]):
            lon = df_tr['LON'][i]
            lat = df_tr['LAT'][i]
            key = str(lon)+":"+str(lat)
            c_label = lonlat_cluster_dict[key]
            if c_label == cluster_label:
                #attr = " ".join([str(c_label),str(df_tr['ENTERTAINMENT'][i]),str(df_tr['SHOPPING'][i])])
                attr = " ".join([str(c_label),str(df_tr['INCOME'][i]),str(df_tr['ENTERTAINMENT'][i]),str(df_tr['BABY'][i]),str(df_tr['SHOPPING'][i])])
                if attr not in dict_tr_x.keys():
                    sub_tr_x = []
                    sub_tr_y = []
                    sub_tr_x.append(tr_x[i])
                    sub_tr_y.append(tr_y[i])
                    dict_tr_x[attr] = sub_tr_x
                    dict_tr_y[attr] = sub_tr_y
                else:
                    sub_tr_x = dict_tr_x[attr]
                    sub_tr_y = dict_tr_y[attr]
                    sub_tr_x.append(tr_x[i])
                    sub_tr_y.append(tr_y[i])
                    dict_tr_x[attr] = sub_tr_x
                    dict_tr_y[attr] = sub_tr_y

    print('1. begin train.....')
    dic_rf = {}
    rf = RandomForestClassifier(
            n_estimators = 100,
            max_depth = 11,
            min_samples_split =2,
            bootstrap =True,
            warm_start = True,
            max_features = 'sqrt',
            criterion='entropy',
            class_weight = 'balanced',
            n_jobs = -1
            )
    clf_svm = SVC()
    total_count = 0
    for attr,sub_tr_x in dict_tr_x.items():
        sub_tr_y = dict_tr_y[attr]
        total_count += len(sub_tr_x)
        print(attr+":\t"+str(len(sub_tr_x)))
        rf = rf.fit(np.array(sub_tr_x),np.array(sub_tr_y))
        #clf_svm = clf_svm.fit(np.array(sub_tr_x),np.array(sub_tr_y))
        dic_rf[attr] = rf
    print(str(total_count))
    print('rf numbers:\t'+str(len(dic_rf)))
    print('2. begin predict.....')
    te_pred_list = []
    for i in range(te_x.shape[0]):
        lon = df_te['LON'][i]
        lat = df_te['LAT'][i]
        key = str(lon)+":"+str(lat)
        c_label = lonlat_cluster_dict[key]
        attr = " ".join([str(c_label),str(df_te['INCOME'][i]),str(df_te['ENTERTAINMENT'][i]),str(df_te['BABY'][i]),str(df_te['SHOPPING'][i])])
        #attr = " ".join([str(c_label),str(df_tr['ENTERTAINMENT'][i]),str(df_tr['SHOPPING'][i])])
        print('te idx:\t'+str(i)+'\t'+attr)
        if attr in dict_tr_x.keys():
            sub_tr_x = dict_tr_x[attr]
            sub_tr_y = dict_tr_y[attr]
            if len(sub_tr_x) == 1:
                te_pred_list.append(sub_tr_y[0])
                continue
            clf = dic_rf[attr]
            te_pred = clf.predict(te_x[i])
            te_pred_list.append(te_pred[0])
        else:
            most_similarity_key = findMostSimilaritySameUserAttr(dict_tr_x,attr)
            sub_tr_x = dict_tr_x[most_similarity_key]
            sub_tr_y = dict_tr_y[most_similarity_key]
            if len(sub_tr_x) == 1:
                te_pred_list.append(sub_tr_y[0])
                continue
            clf = dic_rf[most_similarity_key]
            te_pred = clf.predict(te_x[i])
            te_pred_list.append(te_pred[0])
    
    save_results(result_csv_path,te_pred_list)
    #    for i in range(te_x.shape[0]):
    #        lon = df_te['LON'][i]
    #        lat = df_te['LAT'][i]
    #        key = str(lon)+":"+str(lat)
    #        c_label = lonlat_cluster_dict[key]
    #        attr = " ".join([str(df_te['INCOME'][i]),str(df_te['ENTERTAINMENT'][i]),str(df_te['BABY'][i]),str(df_te['SHOPPING'][i])])
    #        if c_label == cluster_label:
    #            if attr not in te_attr_dict.keys():
    #                te_attr_dict[attr] = 1
    #            else:
    #                te_attr_dict[attr] += 1

    #            if attr not in tr_attr_dict.keys():
    #                IsAllExists = False
    #    print(str(IsAllExists))
    #    print(str(cluster_label)+' Train set...')
    #    for k,v in tr_attr_dict.items():
    #        print(k+":\t"+str(v))
    #    print(str(cluster_label)+' Test set...')
    #    for k,v in te_attr_dict.items():
    #        print(k+":\t"+str(v))

    #    print('\n\n\n')


if __name__=='__main__':
    start = time()
    run()
    end = time()
    print('Time:\t'+str((end-start)/3600)+' Hours')
