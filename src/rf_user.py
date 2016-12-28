#! /usr/bin/env python

from time import time
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans


from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

from utils import load_data,save_results
from config import TRAIN,TEST,SHOP_PROFILE
from config import Features
from config import CLUSTER_DICT

from location import run_cluster_model
from location import run_simple_model

from type_rf import predict_type

result_csv_path = 'result/fusai_user_rf_20161216_28.csv'




def run():
    te_x = load_data(TEST,False)
    df_te = pd.read_csv(TEST,header=0)
    
    te_predType_list = predict_type()
    te_pred_list = []
    for i in range(te_x.shape[0]):
        cur_pred_type = te_predType_list[i]
        if cur_pred_type == 0:
            te_pred_list.append(0)
            continue
        income = str(df_te['INCOME'][i])
        entertainment = str(df_te['ENTERTAINMENT'][i])
        baby = str(df_te['BABY'][i])
        shopclass = str(df_te['SHOPPING'][i])
        key = income+" "+entertainment+" "+baby+" "+shopclass
        lon = df_te['LON'][i]
        lat = df_te['LAT'][i]
        num_clusters = CLUSTER_DICT[key]
        if num_clusters > 1:
            cur_te_pred = run_cluster_model(key,lon,lat,te_x[i],num_clusters)
        else:
            cur_te_pred = run_simple_model(key,te_x[i])
        te_pred_list.append(cur_te_pred)
        print("te_idx:"+str(i))

    print(str(len(te_pred_list)))
    save_results(result_csv_path,te_pred_list)
    
    


    #tr_x,tr_y = load_data(TRAIN,True)
    #df_tr = pd.read_csv(TRAIN,header=0)
    #tr_x_dic = {}
    #tr_y_dic = {}

    #tr_lonlat_list = []
    #for i in range(tr_x.shape[0]):
    #    attr = [ str(df_tr['INCOME'][i]),str(df_tr['ENTERTAINMENT'][i]), str(df_tr['BABY'][i]),str(df_tr['SHOPPING'][i]) ]
    #    key = " ".join(attr)
    #    if key not in tr_x_dic.keys():
    #        sub_tr_x = []
    #        sub_tr_y = []
    #        sub_tr_x.append(tr_x[i])
    #        sub_tr_y.append(tr_y[i])
    #        tr_x_dic[key] = sub_tr_x
    #        tr_y_dic[key] = sub_tr_y
    #    else:
    #        sub_tr_x = tr_x_dic[key]
    #        sub_tr_y = tr_y_dic[key]
    #        sub_tr_x.append(tr_x[i])
    #        sub_tr_y.append(tr_y[i])
    #        tr_x_dic[key] = sub_tr_x
    #        tr_y_dic[key] = sub_tr_y
    #
    #print('key:\t'+str(len(tr_x_dic)))
    #print('key:\t'+str(len(tr_y_dic)))

    #rf = RandomForestClassifier(
    #        n_estimators = 150,
    #        max_depth = 11,
    #        min_samples_split =2,
    #        bootstrap =True,
    #        warm_start = True,
    #        max_features = 'sqrt',
    #        criterion='entropy',
    #        class_weight = 'balanced',
    #        n_jobs = -1
    #        )
    #rf_dic = {}
    #avg_score = 0
    #count = 0
    #for key,sub_tr_x in tr_x_dic.items():
    #    sub_tr_x = np.array(sub_tr_x)
    #    sub_tr_y = np.array(tr_y_dic[key])
    #    if sub_tr_x.shape[0] <= 100:
    #        continue
    #    cv = 10
    #    print(key +":"+str(sub_tr_x.shape[0]))
    #    scores = cross_val_score(rf,sub_tr_x,sub_tr_y,cv=10,scoring='f1_weighted')
    #    print(str(scores))
    #    avg_score += sum(scores)/cv
    #    count += 1
    #    print('avg scores:\t'+str(sum(scores)/cv))
    #    print('\n\n')
    #print('total AVG scroe:\t'+str(avg_score/count))



    #print(str(len(te_pred)))
    #save_results(result_csv_path,te_pred)


if __name__=='__main__':
    start = time()
    run()
    end = time()
    print('Time:\t'+str((end-start)/3600)+' Hours')
