#! /usr/bin/env python

from config import SHOP_PROFILE
from config import USER_PROFILE
from config import TRAIN
from config import TEST

from config import MYTEST

from config import USER_TRACE

from config import NUM_NEAREST_SHOPS
from config import NUM_NEAREST_USERS

from config import NEIGH_CLASS_SIMILARITY
from config import NEIGH_INCOME_SIMILARITY
from config import NEIGH_ENTERTAINMENT_SIMILARITY
from config import NEIGH_BABY_SIMILARITY


from config import DESTINATION_NUM_NEAREST_SHOPS



from utils import caldistance,caltime
from utils import get_hourAndweek
from utils import IsAwayShopCenter

from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib

import numpy as np
import pandas as pd
from time import time


def merge_shop_file():
    f_shop = open(SHOP_PROFILE,'r')
    last_line = ""
    cur_line = ""
    line_list = []
    for line in f_shop:
        l = line.strip().split(',')
        if len(l) == 5:
            line_list.append(line)
        elif len(l) == 4:
            last_line = line
        else:
            cur_line = line
            line_list.append(last_line.strip()+cur_line)
            last_line = ""
    
    f_shop.close()
    f_shop = open(SHOP_PROFILE,'w')
    for line in line_list:
        f_shop.write(line.strip()+'\n')
    f_shop.close()




##### get person topk nearest_shops by his/her position and time
def get_positions(csv_path):
    df = pd.read_csv(csv_path,header = 0)
    df_trace = pd.read_csv(USER_TRACE,header = 0)
    
    lon_list = []
    lat_list = []
    duration_list = []
    durationLevel_list = []
    for i in range(len(df)):
        uid = df['USERID'][i]
        utime = df['ARRIVAL_TIME'][i]
        ###### find a nearest time of this uid ########
        query_trace_df = df_trace[(df_trace['USERID']==uid) & (df_trace['BEGIN_TIME']==utime)]
        nearest_lon = query_trace_df['STARTLONGITUDE'].values[0]
        nearest_lat = query_trace_df['STARTLATITUDE'].values[0]
        nearest_duration = float(query_trace_df['DURATION'].values[0])
        lon_list.append(nearest_lon)
        lat_list.append(nearest_lat)
        duration_list.append(nearest_duration)
        if nearest_duration > 27:
            durationLevel_list.append(2)
        elif nearest_duration < 8:
            durationLevel_list.append(0)
        else:
            durationLevel_list.append(1)
    #print('len:'+str(len(lon_list))+','+str(len(lat_list)))
    df['LON'] = lon_list
    df['LAT'] = lat_list
    df['DURATION'] = duration_list
    df['DURATION_LEVEL'] = durationLevel_list
    df.to_csv(csv_path,index=False)


def get_nearest_shops():
    df_train = pd.read_csv(TRAIN,header=0)
    df_test = pd.read_csv(TEST,header=0)
    df_shop = pd.read_csv(SHOP_PROFILE,header=0)

    train_lonlat_list = np.array(df_train.loc[:,['LON','LAT']])
    test_lonlat_list = np.array(df_test.loc[:,['LON','LAT']])
    shop_lonlat_list = np.array(df_shop.loc[:,['LONGITUDE','LATITUDE']])
    
    neigh = NearestNeighbors(NUM_NEAREST_SHOPS,0.4,metric='euclidean',algorithm='kd_tree',n_jobs=-1)
    neigh.fit(shop_lonlat_list)
    joblib.dump(neigh,'weights/neigh.m')
    
    tr_neighshop_odistance_idxs = np.array(neigh.kneighbors(train_lonlat_list))
    te_neighshop_odistance_idxs = np.array(neigh.kneighbors(test_lonlat_list))

    tr_neighshop_id_list = []
    tr_neighshop_odist_list = []
    tr_neighshop_lonlat_list = []
    tr_neighshop_class_list = []

    tr_distance_centerlonlat_list = []
    tr_direction_list = []
    tr_IsAway_list = []
    for i in range(len(df_train)):
        cur_neigh_idx = np.array(tr_neighshop_odistance_idxs[1,i,:],dtype=int)
        cur_neigh_odist = list(tr_neighshop_odistance_idxs[0,i,:])
        
        cur_neighshop_id_list = []
        cur_neighshop_lonlat_list = []
        cur_neighshop_class_list = []

        center_neighshop_lon = 0
        center_neighshop_lat = 0
        for j in range(len(cur_neigh_odist)):
            cur_neighshop_id_list.append(int(df_shop['ID'][cur_neigh_idx[j]]))
            cur_neighshop_lonlat_list.append(float(df_shop['LONGITUDE'][cur_neigh_idx[j]]))
            cur_neighshop_lonlat_list.append(float(df_shop['LATITUDE'][cur_neigh_idx[j]]))
            cur_neighshop_class_list.append(int(df_shop['CLASSIFICATION'][cur_neigh_idx[j]]))
            center_neighshop_lon += float(df_shop['LONGITUDE'][cur_neigh_idx[j]])
            center_neighshop_lat += float(df_shop['LATITUDE'][cur_neigh_idx[j]])
        tr_neighshop_id_list.append(cur_neighshop_id_list)
        tr_neighshop_odist_list.append(cur_neigh_odist)
        tr_neighshop_lonlat_list.append(cur_neighshop_lonlat_list)
        tr_neighshop_class_list.append(cur_neighshop_class_list)
        center_neighshop_lon = center_neighshop_lon/NUM_NEAREST_SHOPS
        center_neighshop_lat = center_neighshop_lat/NUM_NEAREST_SHOPS
        cur_lon = df_train['LON'][i]
        cur_lat = df_train['LAT'][i]
        dest_lon = df_train['DESTINATION_LON'][i]
        dest_lat = df_train['DESTINATION_LAT'][i]
        cur_center_distance,cur_direction = caldistance(cur_lon,cur_lat,center_neighshop_lon,center_neighshop_lat)
        tr_IsAway_list.append(IsAwayShopCenter(cur_lon,cur_lat,center_neighshop_lon,center_neighshop_lat,dest_lon,dest_lat))
        tr_distance_centerlonlat_list.append(cur_center_distance)
        tr_direction_list.append(cur_direction)


    te_neighshop_id_list = []
    te_neighshop_odist_list = []
    te_neighshop_lonlat_list = []
    te_neighshop_class_list = []
    te_distance_centerlonlat_list = []
    te_direction_list = []
    te_IsAway_list = []

    for i in range(len(df_test)):
        cur_neigh_idx = np.array(te_neighshop_odistance_idxs[1,i,:],dtype=int)
        cur_neigh_odist = list(te_neighshop_odistance_idxs[0,i,:])
        cur_neighshop_id_list = []
        cur_neighshop_lonlat_list = []
        cur_neighshop_class_list = []
        center_neighshop_lon = 0
        center_neighshop_lat = 0
        for j in range(len(cur_neigh_odist)):
            cur_neighshop_id_list.append(int(df_shop['ID'][cur_neigh_idx[j]]))
            cur_neighshop_lonlat_list.append(float(df_shop['LONGITUDE'][cur_neigh_idx[j]]))
            cur_neighshop_lonlat_list.append(float(df_shop['LATITUDE'][cur_neigh_idx[j]]))
            cur_neighshop_class_list.append(int(df_shop['CLASSIFICATION'][cur_neigh_idx[j]]))
            center_neighshop_lon += float(df_shop['LONGITUDE'][cur_neigh_idx[j]])
            center_neighshop_lat += float(df_shop['LATITUDE'][cur_neigh_idx[j]])
        te_neighshop_id_list.append(cur_neighshop_id_list)
        te_neighshop_odist_list.append(cur_neigh_odist)
        te_neighshop_lonlat_list.append(cur_neighshop_lonlat_list)
        te_neighshop_class_list.append(cur_neighshop_class_list)
        center_neighshop_lon = center_neighshop_lon/NUM_NEAREST_SHOPS
        center_neighshop_lat = center_neighshop_lat/NUM_NEAREST_SHOPS
        cur_lon = df_test['LON'][i]
        cur_lat = df_test['LAT'][i]
        dest_lon = df_test['DESTINATION_LON'][i]
        dest_lat = df_test['DESTINATION_LAT'][i]
        cur_center_distance,cur_direction = caldistance(cur_lon,cur_lat,center_neighshop_lon,center_neighshop_lat)
        te_IsAway_list.append(IsAwayShopCenter(cur_lon,cur_lat,center_neighshop_lon,center_neighshop_lat,dest_lon,dest_lat))
        te_distance_centerlonlat_list.append(cur_center_distance)
        te_direction_list.append(cur_direction)
    

    df_train['NEIGHSHOP_ID'] = tr_neighshop_id_list
    df_train['NEIGHSHOP_LONLAT'] = tr_neighshop_lonlat_list
    df_train['NEIGHSHOP_DISTANCE'] = tr_neighshop_odist_list
    df_train['NEIGHSHOP_CLASS'] = tr_neighshop_class_list
    df_train['NEIGHSHOP_CENTER_DISTANCE'] = tr_distance_centerlonlat_list
    df_train['DIRECTION'] = tr_direction_list
    df_train['ISAWAYCENTER'] = tr_IsAway_list

    df_test['NEIGHSHOP_ID'] = te_neighshop_id_list
    df_test['NEIGHSHOP_LONLAT'] = te_neighshop_lonlat_list
    df_test['NEIGHSHOP_DISTANCE'] = te_neighshop_odist_list
    df_test['NEIGHSHOP_CLASS'] = te_neighshop_class_list
    df_test['NEIGHSHOP_CENTER_DISTANCE'] = te_distance_centerlonlat_list
    df_test['DIRECTION'] = te_direction_list
    df_test['ISAWAYCENTER'] = te_IsAway_list




    df_train.to_csv(TRAIN,index=False)
    df_test.to_csv(TEST,index=False)
    
    #df_tr_x = pd.read_csv('../new/TRAIN_NEIGHUSER_SHOPID.csv',header=0)
    #df_te_x = pd.read_csv('../new/TEST_NEIGHUSER_SHOPID.csv',header=0)
    #df_tr_x['NEIGHSHOP_ID'] = tr_neighshop_id_list
    #df_te_x['NEIGHSHOP_ID'] = te_neighshop_id_list
    #df_tr_x.to_csv('../new/TRAIN_NEIGHUSER_SHOPID.csv',index=False)
    #df_te_x.to_csv('../new/TEST_NEIGHUSER_SHOPID.csv',index=False)

def get_user_shop_similarity(csv_path):
    df = pd.read_csv(csv_path,header=0)
    class_similarity_list = []
    income_similarity_list = []
    entertainment_similarity_list = []
    baby_similarity_list = []

    total_similarity_list = []
    for i in range(len(df)):
        #print(str(i))
        cur_user_class_idx = int(df['SHOPPING'][i]) - 1
        cur_user_income_idx = int(df['INCOME'][i]) - 1
        cur_user_entertainment_idx = int(df['ENTERTAINMENT'][i]) - 1
        cur_user_baby_idx = int(df['BABY'][i])
        cur_neigh_shops_class = list(eval(df['NEIGHSHOP_CLASS'][i]))

        cur_class_similarity = []
        cur_income_similarity = []
        cur_entertainment_similarity = []
        cur_baby_similarity = []
        cur_total_similarity = []

        for shop_class_idx in cur_neigh_shops_class:
            cur_class_similarity.append(NEIGH_CLASS_SIMILARITY[cur_user_class_idx][shop_class_idx-1])
            cur_income_similarity.append(NEIGH_INCOME_SIMILARITY[cur_user_income_idx][shop_class_idx-1])
            cur_entertainment_similarity.append(NEIGH_ENTERTAINMENT_SIMILARITY[cur_user_entertainment_idx][shop_class_idx-1])
            cur_baby_similarity.append(NEIGH_BABY_SIMILARITY[cur_user_baby_idx][shop_class_idx-1])
            cur_total_similarity.append(
                    NEIGH_CLASS_SIMILARITY[cur_user_class_idx][shop_class_idx-1]*0.6
                    + NEIGH_INCOME_SIMILARITY[cur_user_income_idx][shop_class_idx-1]*0.1
                    + NEIGH_ENTERTAINMENT_SIMILARITY[cur_user_entertainment_idx][shop_class_idx-1]*0.2
                    + NEIGH_BABY_SIMILARITY[cur_user_baby_idx][shop_class_idx-1]*0.1)
        class_similarity_list.append(cur_class_similarity)
        income_similarity_list.append(cur_income_similarity)
        entertainment_similarity_list.append(cur_entertainment_similarity)
        baby_similarity_list.append(cur_baby_similarity)
        total_similarity_list.append(cur_total_similarity)
    df['NEIGH_CLASS_SIMILARITY'] = class_similarity_list
    df['NEIGH_INCOME_SIMILARITY'] = income_similarity_list
    df['NEIGH_ENTERTAINMENT_SIMILARITY'] = entertainment_similarity_list
    df['NEIGH_BABY_SIMILARITY'] = baby_similarity_list
    df['NEIGH_SIMILARITY'] = total_similarity_list
    df.to_csv(csv_path,index=False)



def get_userattributs(csv_path):
    df = pd.read_csv(csv_path,header=0)
    df_user = pd.read_csv(USER_PROFILE,header=0)
    df = pd.merge(df,df_user,on='USERID')
    df.to_csv(csv_path,index=False)



def get_timeFeature(csv_path):
    df = pd.read_csv(csv_path,header=0)
    hour_list = []
    week_list = []
    date_list = []
    for i in range(len(df)):
        cur_time = df['ARRIVAL_TIME'][i]
        hour ,week = get_hourAndweek(str(cur_time))
        hour_list.append(hour)
        week_list.append(week)
        date = int(str(cur_time)[6:8])
        date_list.append(date)
    df['HOUR'] = hour_list
    df['WEEK'] = week_list
    df['DATE'] = date_list
    df.to_csv(csv_path,index=False)


def calculateDistanceTime(df):
    total_distance = 0
    total_time = 0
    idxs = df.index
    for i in range(1,len(df)):
        last_idx = idxs[i-1]
        cur_idx = idxs[i]
        last_lon = df['STARTLONGITUDE'][last_idx]
        last_lat = df['STARTLATITUDE'][last_idx]
        last_date = df['BEGIN_TIME'][last_idx]
        cur_lon = df['STARTLONGITUDE'][cur_idx]
        cur_lat = df['STARTLATITUDE'][cur_idx]
        cur_date = df['BEGIN_TIME'][cur_idx]
        total_distance +=(caldistance(last_lon,last_lat,cur_lon,cur_lat))[0]
        total_time += caltime(str(last_date),str(cur_date))
    return total_distance,total_time


def get_distanceAndtime(csv_path):
    df_trace = pd.read_csv(USER_TRACE,header=0)
    df = pd.read_csv(csv_path,header=0)
    passDistance_list = []
    passTime_list = []

    for i in range(len(df)):
        cur_uid = df['USERID'][i]
        cur_time = df['ARRIVAL_TIME'][i]
        start_time = (cur_time/1000000) * 1000000
        query_df = df_trace[(df_trace['USERID'] == cur_uid) & (df_trace['BEGIN_TIME'] <= cur_time) & (df_trace['BEGIN_TIME']>=start_time)]
        if len(query_df) == 1:
            passDistance_list.append(0)
            passTime_list.append(0)
        else:
            cur_passDistance,cur_passTime = calculateDistanceTime(query_df)
            passDistance_list.append(cur_passDistance)
            passTime_list.append(cur_passTime)
    df['PASSDISTANCE'] = passDistance_list
    df['PASSTIME'] = passTime_list
    df.to_csv(csv_path,index=False)

def get_DestinationFeature(csv_path):
    df = pd.read_csv(csv_path,header=0)
    df_shop = pd.read_csv(SHOP_PROFILE,header=0)
    df_trace = pd.read_csv(USER_TRACE,header=0)
    
    shop_lonlat_list = np.array(df_shop.loc[:,['LONGITUDE','LATITUDE']])
    neigh_model = NearestNeighbors(DESTINATION_NUM_NEAREST_SHOPS,0.5,metric='euclidean',algorithm='kd_tree',n_jobs=-1)
    neigh_model.fit(shop_lonlat_list)

    destination_lon_list = []
    destination_lat_list = []

    for i in range(len(df)):
        cur_uid = df['USERID'][i]
        cur_time = df['ARRIVAL_TIME'][i]
        
        end_time = (cur_time / 1000000 )*1000000 + 240000
        query_df = df_trace[(df_trace['USERID'] == cur_uid) & (df_trace['BEGIN_TIME'] >= cur_time) & (df_trace['BEGIN_TIME'] < end_time)]
        lon_list = list(query_df['STARTLONGITUDE'])
        lat_list = list(query_df['STARTLATITUDE'])
        destination_lon_list.append(lon_list[-1])
        destination_lat_list.append(lat_list[-1])

    df['DESTINATION_LON'] = destination_lon_list
    df['DESTINATION_LAT'] = destination_lat_list
    destination_lonlat_list = np.array(df.loc[:,['DESTINATION_LON','DESTINATION_LAT']])
    destination_neighshop_odistance_idxs = np.array(neigh_model.kneighbors(destination_lonlat_list))

    destination_nearest_shop_lonlat_list = []
    destination_nearest_shop_distance_list = []
    destination_nearest_shop_id_list = []
    destination_nearest_shop_class_list = []

    for i in range(len(df)):
        cur_neigh_idx = np.array(destination_neighshop_odistance_idxs[1,i,:],dtype=int)
        cur_neigh_odist = list(destination_neighshop_odistance_idxs[0,i,:])
        cur_neighshop_id_list = []
        cur_neighshop_lonlat_list = []
        cur_neighshop_class_list = []
        for j in range(len(cur_neigh_odist)):
            cur_neighshop_id_list.append(int(df_shop['ID'][cur_neigh_idx[j]]))
            cur_neighshop_lonlat_list.append(float(df_shop['LONGITUDE'][cur_neigh_idx[j]]))
            cur_neighshop_lonlat_list.append(float(df_shop['LATITUDE'][cur_neigh_idx[j]]))
            cur_neighshop_class_list.append(int(df_shop['CLASSIFICATION'][cur_neigh_idx[j]]))
        destination_nearest_shop_id_list.append(cur_neighshop_id_list)
        destination_nearest_shop_distance_list.append(cur_neigh_odist)
        destination_nearest_shop_lonlat_list.append(cur_neighshop_lonlat_list)
        destination_nearest_shop_class_list.append(cur_neighshop_class_list)

    df['DESTINATION_NEIGHSHOP_ID']  = destination_nearest_shop_id_list
    df['DESTINATION_NEIGHSHOP_LONLAT'] = destination_nearest_shop_lonlat_list
    df['DESTINATION_NEIGHSHOP_DISTANCE'] = destination_nearest_shop_distance_list
    df['DESTINATION_NEIGHSHOP_CLASS'] =  destination_nearest_shop_class_list
    df.to_csv(csv_path,index=False)


def get_StartPointFeature(csv_path):
    df = pd.read_csv(csv_path,header=0)
    df_shop = pd.read_csv(SHOP_PROFILE,header=0)
    df_trace = pd.read_csv(USER_TRACE,header=0)
    
    shop_lonlat_list = np.array(df_shop.loc[:,['LONGITUDE','LATITUDE']])
    neigh_model = NearestNeighbors(DESTINATION_NUM_NEAREST_SHOPS,0.5,metric='euclidean',algorithm='kd_tree',n_jobs=-1)
    neigh_model.fit(shop_lonlat_list)

    start_lon_list = []
    start_lat_list = []

    for i in range(len(df)):
        cur_uid = df['USERID'][i]
        cur_time = df['ARRIVAL_TIME'][i]
        
        start_time = (cur_time / 1000000 )*1000000
        query_df = df_trace[(df_trace['USERID'] == cur_uid) & (df_trace['BEGIN_TIME'] <= cur_time) & (df_trace['BEGIN_TIME'] > start_time)]
        lon_list = list(query_df['STARTLONGTITUDE'])
        lat_list = list(query_df['STARTLATITUDE'])
        start_lon_list.append(lon_list[0])
        start_lat_list.append(lat_list[0])

    df['START_LON'] = start_lon_list
    df['START_LAT'] = start_lat_list
    start_lonlat_list = np.array(df.loc[:,['START_LON','START_LAT']])
    start_neighshop_odistance_idxs = np.array(neigh_model.kneighbors(start_lonlat_list))

    start_nearest_shop_lonlat_list = []
    start_nearest_shop_distance_list = []
    start_nearest_shop_id_list = []
    start_nearest_shop_class_list = []

    for i in range(len(df)):
        cur_neigh_idx = np.array(start_neighshop_odistance_idxs[1,i,:],dtype=int)
        cur_neigh_odist = list(start_neighshop_odistance_idxs[0,i,:])
        cur_neighshop_id_list = []
        cur_neighshop_lonlat_list = []
        cur_neighshop_class_list = []
        for j in range(len(cur_neigh_odist)):
            cur_neighshop_id_list.append(int(df_shop['ID'][cur_neigh_idx[j]]))
            cur_neighshop_lonlat_list.append(float(df_shop['LONGITUDE'][cur_neigh_idx[j]]))
            cur_neighshop_lonlat_list.append(float(df_shop['LATITUDE'][cur_neigh_idx[j]]))
            cur_neighshop_class_list.append(int(df_shop['CLASSIFICATION'][cur_neigh_idx[j]]))
        start_nearest_shop_id_list.append(cur_neighshop_id_list)
        start_nearest_shop_distance_list.append(cur_neigh_odist)
        start_nearest_shop_lonlat_list.append(cur_neighshop_lonlat_list)
        start_nearest_shop_class_list.append(cur_neighshop_class_list)

    df['START_NEIGHSHOP_ID']  = start_nearest_shop_id_list
    df['START_NEIGHSHOP_LONLAT'] = start_nearest_shop_lonlat_list
    df['START_NEIGHSHOP_DISTANCE'] = start_nearest_shop_distance_list
    df['START_NEIGHSHOP_CLASS'] =  start_nearest_shop_class_list
    df.to_csv(csv_path,index=False)





def get_trjMostShopTypes(csv_path):
    df = pd.read_csv(csv_path,header=0)
    df_trace = pd.read_csv(USER_TRACE,header=0)
    df_shop = pd.read_csv(SHOP_PROFILE,header=0)
    neigh = joblib.load('weights/neigh.m')

    OccurMostShopType_list = []
    for i in range(len(df)):
        cur_uid = df['USERID'][i]
        cur_time = df['ARRIVAL_TIME'][i]
        start_time = (cur_time / 1000000)*1000000
        end_time = (cur_time / 1000000 )*1000000 + 240000

        query_df = df_trace[(df_trace['USERID'] == cur_uid) & (df_trace['BEGIN_TIME'] >= start_time) & (df_trace['BEGIN_TIME'] < end_time)]
        lon_list = np.array(query_df['STARTLONGITUDE'])
        lat_list = np.array(query_df['STARTLATITUDE'])
        cur_trj = np.vstack([lon_list,lat_list])
        cur_trj = np.transpose(cur_trj)
        curTrj_neighshop_odistance_idxs = np.array(neigh.kneighbors(cur_trj))

        type_dic = {}
        for j in range(cur_trj.shape[0]):
            cur_neigh_idx = np.array(curTrj_neighshop_odistance_idxs[1,j,:],dtype=int)
            for k in range(len(cur_neigh_idx)):
                cur_neighshop_type = int(df_shop['CLASSIFICATION'][cur_neigh_idx[k]])
                if cur_neighshop_type in type_dic.keys():
                    type_dic[cur_neighshop_type] += 1
                else:
                    type_dic[cur_neighshop_type] = 1
        sorted_shoptypes = sorted(type_dic.iteritems(),key=lambda k:k[1],reverse=True)
        OccurMostShopType_list.append(sorted_shoptypes[0][0])
    df['MOSTSHOPTYPE'] = OccurMostShopType_list
    df.to_csv(csv_path,index=False)


def get_neighUserFeature(csv_path,IsTrain):
    df_tr = pd.read_csv(TRAIN,header=0)
    df_shop = pd.read_csv(SHOP_PROFILE,header=0)
    df = pd.read_csv(csv_path,header=0)

    tr_user_lonlat_list = np.array(df_tr.loc[:,['LON','LAT']])
    neigh_model = NearestNeighbors(NUM_NEAREST_USERS+1,0.5,metric='euclidean',algorithm='kd_tree',n_jobs=-1)
    neigh_model.fit(tr_user_lonlat_list)

    user_lonlat_list = np.array(df.loc[:,['LON','LAT']])
    neighuser_odistance_idxs = np.array(neigh_model.kneighbors(user_lonlat_list))

    neighuser_distance_list = []
    neighuser_lonlat_list = []
    neighuser_class_list = []
    neighuser_shoptype_list = []
    neighuser_shopid_list = []

    for i in range(len(df)):
        cur_neigh_idx = np.array(neighuser_odistance_idxs[1,i,1:],dtype=int) if IsTrain else np.array(neighuser_odistance_idxs[1,i,:-1],dtype=int)
        cur_neigh_odist = list(neighuser_odistance_idxs[0,i,1:]) if IsTrain else list(neighuser_odistance_idxs[0,i,:-1])
        
        cur_neighuser_lonlat_list = []
        cur_neighuser_class_list = []
        cur_neighuser_shoptype_list = []
        cur_neighuser_shopid_list = []
        for j in range(len(cur_neigh_idx)):
            cur_neighuser_lonlat_list.append(float(df_tr['LON'][cur_neigh_idx[j]]))
            cur_neighuser_lonlat_list.append(float(df_tr['LAT'][cur_neigh_idx[j]]))
            cur_neighuser_class_list.append(int(df_tr['SHOPPING'][cur_neigh_idx[j]]))
            cur_neighuser_shopid_list.append(int(df_tr['SHOPID'][cur_neigh_idx[j]]))
            query_df = df_shop[(df_shop['ID']==df_tr['SHOPID'][cur_neigh_idx[j]])]
            if len(query_df) == 0:
                cur_neighuser_shoptype_list.append(0)
            else:
                cur_neighuser_shoptype_list.append(list(query_df['CLASSIFICATION'])[0])
        neighuser_distance_list.append(cur_neigh_odist)
        neighuser_lonlat_list.append(cur_neighuser_lonlat_list)
        neighuser_class_list.append(cur_neighuser_class_list)
        neighuser_shoptype_list.append(cur_neighuser_shoptype_list)
        neighuser_shopid_list.append(cur_neighuser_shopid_list)
    
    df['NEIGHUSER_LONLAT'] = neighuser_lonlat_list
    df['NEIGHUSER_DISTANCE'] = neighuser_distance_list
    df['NEIGHUSER_CLASS'] = neighuser_class_list
    df['NEIGHUSER_SHOPID'] = neighuser_shopid_list
    df['NEIGHUSE_SHOPTYPE'] = neighuser_shoptype_list
    df.to_csv(csv_path,index=False)
    #df_tr_x = pd.read_csv('../new/TRAIN_NEIGHUSER_SHOPID.csv',header=0)
    #df_te_x = pd.read_csv('../new/TEST_NEIGHUSER_SHOPID.csv',header=0)
    #df_tr_x['NEIGHUSER_SHOPID'] = neighuser_shopid_list
    #df_te_x['NEIGHUSER_SHOPID'] = neighuser_shopid_list
    #df_tr_x.to_csv('../new/TRAIN_NEIGHUSER_SHOPID.csv',index=False)
    #df_te_x.to_csv('../new/TEST_NEIGHUSER_SHOPID.csv',index=False)


def joinShopFile(csv_path):
    df = pd.read_csv(csv_path,header=0)
    df_shop = pd.read_csv(SHOP_PROFILE,header=0)
    shop_class_list = []
    shop_lon_list = []
    shop_lat_list = []
    for i in range(len(df)):
        #print(str(i))
        shopid = df['SHOPID'][i]
        query_df = df_shop[df_shop['ID']==shopid]
        if len(query_df) != 0:
            shop_class_list.append(df_shop[df_shop['ID']==shopid]['CLASSIFICATION'].values[0])
            shop_lon_list.append(df_shop[df_shop['ID']==shopid]['LONGITUDE'].values[0])
            shop_lat_list.append(df_shop[df_shop['ID']==shopid]['LATITUDE'].values[0])
        else:
            shop_class_list.append(0)
            shop_lon_list.append(100)
            shop_lat_list.append(100)

    df['SHOP_CLASS']=shop_class_list
    df['SHOP_LON']=shop_lon_list
    df['SHOP_LAT']=shop_lat_list
    df.to_csv(csv_path,index=False)



def get_IsUserHasGotoShop(csv_path):
    df_tr = pd.read_csv(TRAIN,header=0)
    df = pd.read_csv(csv_path,header=0)

    gotoshops_list = []
    for i in range(len(df)):
        dic_shops = {}
        uid = df['USERID'][i]
        cur_gotoshops_list = list(df_tr[df_tr['USERID']==uid]['SHOP_CLASS'].values)
        for j in range(len(cur_gotoshops_list)):
            if cur_gotoshops_list[j] != 0:
                if cur_gotoshops_list[j] not in dic_shops.keys():
                    dic_shops[cur_gotoshops_list[j]] = 1
                else:
                    dic_shops[cur_gotoshops_list[j]] += 1
        cur_gotoshops_list = []
        for shoptype in range(1,11):
            if shoptype in dic_shops.keys():
                cur_gotoshops_list.append(dic_shops[shoptype])
            else:
                cur_gotoshops_list.append(0)
        gotoshops_list.append(cur_gotoshops_list)

    df['PASSSHOPTYPE_COUNT'] = gotoshops_list
    df.to_csv(csv_path,index=False)

def get_Neigh_MostShopType(csv_path):
    df = pd.read_csv(csv_path,header=0)
    Mostneighshoptype_list = []
    Mostneighshoptype_count_list = []
    for i in range(len(df)):
        dic = {}
        cur_neighshop_type_list = list(eval(df['NEIGHSHOP_CLASS'][i]))
        for shoptype in cur_neighshop_type_list:
            if shoptype not in dic.keys():
                dic[shoptype] = 1
            else:
                dic[shoptype] += 1
            sorted_dic = sorted(dic.iteritems(),key=lambda x:x[1],reverse=True)
        Mostneighshoptype_list.append(sorted_dic[0][0])
        Mostneighshoptype_count_list.append(sorted_dic[0][1])
    df['NEIGHSHOP_MOSTTYPE'] = Mostneighshoptype_list
    df['NEIGHSHOP_MOSTTYPE_COUNT'] = Mostneighshoptype_count_list
    df.to_csv(csv_path,index=False)


def User_To_EachTypeShop_Count(csv_path):
    df = pd.read_csv(csv_path,header=0)
    df_tr = pd.read_csv(TRAIN,header=0)

    shoptypeCount_list = []
    for i in range(len(df)):
        uid = df['USERID'][i]
        cur_shopclass_list = [0]*10
        shopclass_list = list(df_tr[df_tr['USERID']==uid]['SHOP_CLASS'].values)
        for j in range(len(shopclass_list)):
            if shopclass_list[j] != 0:
                cur_shopclass_list[shopclass_list[j]-1] += 1
        shoptypeCount_list.append(cur_shopclass_list)

    df['SHOPTYPE_COUNT'] = shoptypeCount_list
    df.to_csv(csv_path,index=False)


def getMostOccShopOfNeighUser(csv_path,isTrain=True):
    df = pd.read_csv(csv_path,header=0)
    df_neigh = pd.read_csv('../new/TRAIN_NEIGHUSER_SHOPID.csv',header=0) if isTrain else pd.read_csv('../new/TEST_NEIGHUSER_SHOPID.csv',header=0)
    df_shop = pd.read_csv(SHOP_PROFILE,header=0)
    top_occurr_shopidList = []
    top_orrurr_shoptypeList = []
    for i in range(len(df)):
        neigh_shopids = list(eval(df_neigh['NEIGHUSER_SHOPID'][i]))
        dict_shopid_occurrences = {}
        for j in range(len(neigh_shopids)):
            cur_neighshopid = int(neigh_shopids[j])
            if cur_neighshopid ==0:
                continue
            if cur_neighshopid not in dict_shopid_occurrences.keys():
                dict_shopid_occurrences[cur_neighshopid] = 1
            else:
                dict_shopid_occurrences[cur_neighshopid] += 1
        sorted_list = sorted(dict_shopid_occurrences.iteritems(),key=lambda x:x[1],reverse=True)
        top_occurr_shopidList.append(sorted_list[0][0])

        shoptype = list(df_shop[df_shop['ID']==sorted_list[0][0]]['CLASSIFICATION'].values)[0]
        top_orrurr_shoptypeList.append(shoptype)
    df['NEIGHUSER_MOST_GOTO'] = top_occurr_shopidList
    df['NEIGHUSER_MOST_SHOPTYPE'] = top_orrurr_shoptypeList
    df.to_csv(csv_path,index=False)


def MostRecommendType(csv_path):
    df_tr = pd.read_csv(TRAIN,header=0)
    df = pd.read_csv(csv_path,header=0)
    dict_userkey_shopclass_count = {}

    for i in range(len(df_tr)):
        if df_tr['SHOPID'][i] == 0:
            continue
        income = str(df_tr['INCOME'][i])
        entertainment = str(df_tr['ENTERTAINMENT'][i])
        baby = str(df_tr['BABY'][i])
        shopping = str(df_tr['SHOPPING'][i])
        userkey = income + ' '+entertainment + ' '+baby+' '+shopping
        shopclass = str(df_tr['SHOP_CLASS'][i])
        key = userkey+' '+shopclass
        if key not in dict_userkey_shopclass_count.keys():
            dict_userkey_shopclass_count[key] = 1
        else:
            dict_userkey_shopclass_count[key] += 1

    recommendMostType_list = []
    for i in range(len(df)):
        income = str(df['INCOME'][i])
        entertainment = str(df['ENTERTAINMENT'][i])
        baby = str(df['BABY'][i])
        shopping = str(df['SHOPPING'][i])
        userkey = income+' '+entertainment+' '+baby+' '+shopping

        shoptypeList = range(1,11)
        most_shoptype = 0
        most_count = 0
        for shoptype in shoptypeList:
            key = userkey+" "+str(shoptype)
            if key not in dict_userkey_shopclass_count.keys():
                continue
            else:
                count = dict_userkey_shopclass_count[key]
                if count > most_count:
                    most_count = count
                    most_shoptype = shoptype
        recommendMostType_list.append(most_shoptype)

    df['RECOMMEND_MOST_TYPE'] = recommendMostType_list
    df.to_csv(csv_path,index=False)

def addMF_feature_user_shop(csv_path):
    # df_mf = pd.read_csv('../new/userfeature.csv_k_15_t_20',header=0)  ### best until now
    df_mf = pd.read_csv('../new/14_all_hyk_userfeature.csv.k_30_t_20',header=0)
    df_user = pd.read_csv(USER_PROFILE,header=0)
    df_mf = pd.merge(df_mf,df_user,on='USERID')
    df = pd.read_csv(csv_path,header=0)
    #df.drop('MF_USERFEATURE',axis=1,inplace=True)
    mf_feature_list = []
    n = 30
    for i in range(len(df)):
        uid = (df['USERID'][i])
        query_df = df_mf[df_mf['USERID']==uid]
        if len(query_df) != 0:
            mf_feature_list.append(list(query_df['MF_USERFEATURE'].values)[0])
        else:
            income = df['INCOME'][i]
            entertainment = df['ENTERTAINMENT'][i]
            baby = df['BABY'][i]
            shopping = df['SHOPPING'][i]
            similarity_df = df_mf[(df_mf['INCOME']==income) & (df_mf['ENTERTAINMENT']==entertainment) & (df_mf['BABY']==baby) & (df_mf['SHOPPING']==shopping)]
            if len(similarity_df) !=0:
                similarity_mffeature_list = list(similarity_df['MF_USERFEATURE'].values)
                similarity_mffeature = [0]*n
                for j in range(len(similarity_mffeature_list)):
                    cur_mffeature = list(eval(similarity_mffeature_list[j]))
                    for k in range(n):
                        similarity_mffeature[k] += float(cur_mffeature[k])

                for k in range(n):
                    similarity_mffeature[k] = similarity_mffeature[k]/n
                mf_feature_list.append(str(similarity_mffeature))
            else:
                similarity_mffeature_list = list(df_mf['MF_USERFEATURE'].values)
                similarity_mffeature = [0]*n
                for j in range(len(similarity_mffeature_list)):
                    cur_mffeature = list(eval(similarity_mffeature_list[j]))
                    for k in range(n):
                        similarity_mffeature[k] += float(cur_mffeature[k])
                for k in range(n):
                    similarity_mffeature[k] = similarity_mffeature[k]/n
                mf_feature_list.append(str(similarity_mffeature))
        print(str(i))
    df['MF_USERFEATURE_SHOP'] = mf_feature_list
    df.to_csv(csv_path,index=False)


def addMF_feature(csv_path):
    df_loc_key = pd.read_csv('../new/loction_key.txt_all',header=0)
    df_mf_user = pd.read_csv('../new/14_all_hyk_userfeature.csv.k_20_t_20',header=0)
    df_mf_loc = pd.read_csv('../new/14_all_hyk_productfeature.csv.k_20_t_20',header=0)
    df = pd.read_csv(csv_path,header=0)
    user_mffeature_list = []
    loc_mffeature_list = []
    for i in range(len(df)):
        ###### add user mffeature ####
        uid = int(df['USERID'][i])
        print(str(i)+':'+str(uid))
        query_user_df = df_mf_user[df_mf_user['USERID']==uid]
        cur_user_mffeature = list(query_user_df['MF_USERFEATURE'].values)[0]
        user_mffeature_list.append(cur_user_mffeature)

        #### add loc mffeature ####
        lon = df['LON'][i]
        lat = df['LAT'][i]
        #print(str(lon)+','+str(lat))
        query_df = df_loc_key[(df_loc_key['LON']==lon) & (df_loc_key['LAT']==lat)]

        loc_idx = list(query_df['LOC_IDX'].values)[0]
        cur_loc_feature = list(df_mf_loc[df_mf_loc['LOC_IDX']==loc_idx]['MF_LOCFEATURE'].values)[0]
        loc_mffeature_list.append(cur_loc_feature)


    df['MF_USERFEATURE'] = user_mffeature_list
    df['MF_LOCFEATURE'] = loc_mffeature_list
    df.to_csv(csv_path,index=False)



def addMF_feature_user_shopType(csv_path):
    # df_mf = pd.read_csv('../new/userfeature.csv_k_15_t_20',header=0)  ### best until now
    df_mf = pd.read_csv('../new/shoptype_userfeature_10_t_20.csv',header=0)
    df_user = pd.read_csv(USER_PROFILE,header=0)
    df_mf = pd.merge(df_mf,df_user,on='USERID')
    df = pd.read_csv(csv_path,header=0)
    #df.drop('MF_USERFEATURE',axis=1,inplace=True)
    mf_feature_list = []
    n = 10
    for i in range(len(df)):
        uid = (df['USERID'][i])
        query_df = df_mf[df_mf['USERID']==uid]
        if len(query_df) != 0:
            mf_feature_list.append(list(query_df['MF_USERFEATURE'].values)[0])
        else:
            income = df['INCOME'][i]
            entertainment = df['ENTERTAINMENT'][i]
            baby = df['BABY'][i]
            shopping = df['SHOPPING'][i]
            similarity_df = df_mf[(df_mf['INCOME']==income) & (df_mf['ENTERTAINMENT']==entertainment) & (df_mf['BABY']==baby) & (df_mf['SHOPPING']==shopping)]
            if len(similarity_df) !=0:
                similarity_mffeature_list = list(similarity_df['MF_USERFEATURE'].values)
                similarity_mffeature = [0]*n
                for j in range(len(similarity_mffeature_list)):
                    cur_mffeature = list(eval(similarity_mffeature_list[j]))
                    for k in range(n):
                        similarity_mffeature[k] += float(cur_mffeature[k])

                for k in range(n):
                    similarity_mffeature[k] = similarity_mffeature[k]/n
                mf_feature_list.append(str(similarity_mffeature))
            else:
                similarity_mffeature_list = list(df_mf['MF_USERFEATURE'].values)
                similarity_mffeature = [0]*n
                for j in range(len(similarity_mffeature_list)):
                    cur_mffeature = list(eval(similarity_mffeature_list[j]))
                    for k in range(n):
                        similarity_mffeature[k] += float(cur_mffeature[k])
                for k in range(n):
                    similarity_mffeature[k] = similarity_mffeature[k]/n
                mf_feature_list.append(str(similarity_mffeature))
        print(str(i))
    df['MF_USERFEATURE_SHOPTYPE'] = mf_feature_list
    df.to_csv(csv_path,index=False)



def addLDA_feature(csv_path):
    df_loc_key = pd.read_csv('../new/loction_key.txt_all',header=0)
    df_mf_loc = pd.read_csv('../new/lda_productFeature.txt',header=0)
    df = pd.read_csv(csv_path,header=0)
    loc_mffeature_list = []
    for i in range(len(df)):
        ###### add user mffeature ####
       # uid = int(df['USERID'][i])
       # print(str(i)+':'+str(uid))
       # query_user_df = df_mf_user[df_mf_user['USERID']==uid]
       # cur_user_mffeature = list(query_user_df['USER_SVDFEATURE'].values)[0]
       # user_mffeature_list.append(cur_user_mffeature)

        #### add loc mffeature ####
        lon = df['LON'][i]
        lat = df['LAT'][i]
        #print(str(lon)+','+str(lat))
        query_df = df_loc_key[(df_loc_key['LON']==lon) & (df_loc_key['LAT']==lat)]

        loc_idx = list(query_df['LOC_IDX'].values)[0]
        cur_loc_feature = list(df_mf_loc[df_mf_loc['LOC_IDX']==loc_idx]['LDA_LOCFEATURE'].values)[0]
        loc_mffeature_list.append(cur_loc_feature)


    #df['USER_SVDFEATURE'] = user_mffeature_list
    df['LDA_LOCFEATURE'] = loc_mffeature_list
    df.to_csv(csv_path,index=False)

def run():
    #merge_shop_file()


    #get_positions(TRAIN)
    #get_positions(MYTEST)
    #get_positions(TEST)

    #get_nearest_shops()
    #get_user_shop_similarity(TRAIN)
    #get_user_shop_similarity(MYTEST)
    #get_user_shop_similarity(TEST)

    #get_userattributs(TRAIN)
    #get_userattributs(TEST)
    #get_userattributs(MYTEST)
    #
    #get_timeFeature(TRAIN)
    #get_timeFeature(MYTEST)
    #get_timeFeature(TEST)

    #get_distanceAndtime(TRAIN)
    #get_distanceAndtime(MYTEST)
    #get_distanceAndtime(TEST)

    #get_DestinationFeature(TRAIN)
    #get_DestinationFeature(MYTEST)
    #get_DestinationFeature(TEST)

    #####get_StartPointFeature(TRAIN)
    #####get_StartPointFeature(TEST)
    ####
    ##get_trjMostShopTypes(TRAIN)
    ##get_trjMostShopTypes(MYTEST)
    ##get_trjMostShopTypes(TEST)

    #get_neighUserFeature(TRAIN,True)
    #get_neighUserFeature(MYTEST,False)
    #get_neighUserFeature(TEST,False)

    #joinShopFile(TRAIN)
    #joinShopFile(MYTEST)

    #get_IsUserHasGotoShop(TRAIN)
    #get_IsUserHasGotoShop(TEST)

    #get_Neigh_MostShopType(TRAIN)
    #get_Neigh_MostShopType(TEST)
    
    #User_To_EachTypeShop_Count(TRAIN)
    #User_To_EachTypeShop_Count(TEST)

   # getMostOccShopOfNeighUser(TRAIN,True)
    #getMostOccShopOfNeighUser(TEST,False)

   # MostRecommendType(TRAIN)
   # MostRecommendType(TEST)
    
    #addMF_feature(TRAIN)
    #addMF_feature(TEST)

    #addMF_feature_user_shop(TRAIN)
    #addMF_feature_user_shop(TEST)
   
   addMF_feature_user_shopType(TRAIN)
   addMF_feature_user_shopType(TEST)


    #addLDA_feature(TRAIN)
    #addLDA_feature(TEST)

if __name__=='__main__':
    start = time()
    run()
    end = time()
    print('Time:\t'+str((end - start)/3600) +' Hours')
