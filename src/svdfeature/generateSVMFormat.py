#! /usr/bin/env python

import pandas as pd
import numpy as np
from time import time
sys.path.append('..')
from config import List_Features,Common_Features
from config import TRAIN,TEST,MYTEST
from config import NUM_NEAREST_SHOPS
from config import NUM_NEAREST_USERS
from config import DESTINATION_NUM_NEAREST_SHOPS

TRAIN_SVMFORMAT = 'TRAIN_SVMFORMAT.txt'
TEST_SVMFORMAT = 'TEST_SVMFORMAT.txt'

TRAIN_LRFORMAT = '../new/TRAIN_LRFORMAT.txt'
TEST_LRFORMAT = '../new/TEST_LRFORMAT.txt'

NUM_USER = NUM_NEAREST_SHOPS* 7 + 3*NUM_NEAREST_USERS+len(Common_Features)
NUM_SHOP = 3



def get_UserFeature(csv_path):
    df = pd.read_csv(csv_path,header=0)
    ufeature_list = []
    for i in range(len(df)):
        cur_gfeature = []
        uid = int(df['USERID'][i])
        for j in range(len(List_Features)):
            col_name = List_Features[j]
            cur_feature_list = list(eval(df[col_name][i]))
            cur_gfeature += cur_feature_list

        for j in range(len(Common_Features)):
            col_name = Common_Features[j]
            cur_feature = df[col_name][i]
            cur_gfeature.append(cur_feature)
        print(str(len(cur_gfeature)))
        cur_gfeature_str = ""
        for j in range(len(cur_gfeature)):
            cur_gfeature_str += str(uid)+":"+str(cur_gfeature[j])+" "
            #cur_gfeature_str += str(cur_gfeature[j])+" "
        ufeature_list.append(cur_gfeature_str[:-1])
    return ufeature_list


def get_ShopFeature(csv_path):
    df = pd.read_csv(csv_path,header=0)
    shopfeature_list = []
    for i in range(len(df)):
        shopid = int(df['SHOPID'][i])
        shopclass = df['SHOP_CLASS'][i]
        shoplon = df['SHOP_LON'][i]
        shoplat = df['SHOP_LAT'][i]
        shopfeature_str = str(shopid)+":"+str(shopclass)+" "+str(shopid)+":"+str(shoplon)+" "+str(shopid)+":"+str(shoplat)
        #shopfeature_str =str(shopclass)+" "+str(shoplon)+" "+str(shoplat)
        shopfeature_list.append(shopfeature_str)
    return shopfeature_list






def generate_svmFile(csv_path,IsTrain=True):
    df = pd.read_csv(csv_path,header=0)
    user_feature_list = get_UserFeature(csv_path)
    shop_feature_list = get_ShopFeature(csv_path)
    
    f_out = open(TRAIN_SVMFORMAT,'w') if IsTrain else open(TEST_SVMFORMAT,'w')
    for i in range(len(df)):
        target = 0 if df['SHOPID'][i]==0 else 1
        svm_str = str(target)+" 0 "+str(NUM_USER)+" "+str(NUM_SHOP)+" "+user_feature_list[i]+" "+shop_feature_list[i]+'\n'
        f_out.write(svm_str)
    f_out.close()

def generate_LRFile(csv_path,IsTrain=True):
    df = pd.read_csv(csv_path,header=0)
    user_feature_list = get_UserFeature(csv_path)
    shop_feature_list = get_ShopFeature(csv_path)
    
    f_out = open(TRAIN_LRFORMAT,'w') if IsTrain else open(TEST_LRFORMAT,'w')
    for i in range(len(df)):
        target = 0 if df['SHOPID'][i]==0 else 1
        lr_str = str(target)+" "+user_feature_list[i]+" "+shop_feature_list[i]+'\n'
        f_out.write(lr_str)
    f_out.close()





def run():
    generate_svmFile(TRAIN,True)
    generate_svmFile(MYTEST,False)

    #generate_LRFile(TRAIN,True)
    #generate_LRFile(MYTEST,False)

if __name__=='__main__':
    START = time()
    run()
    END = time()
    print('Cost Time:\t'+str((END-START)/3600)+' Hours')
