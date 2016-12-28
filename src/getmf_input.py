

TRAIN_PATH='../new/TRAIN_NEW.csv'

import pandas as pd



if __name__=='__main__':
    #df_tr = pd.read_csv('../new/TRAIN_NEW.csv',header=0)
    #df_te = pd.read_csv('../new/TEST_NEW.csv',header=0)

    #n = 0
    #dic_loc_key = {}
    #for i in range(len(df_tr)):
    #    lon = str(df_tr['LON'][i])
    #    lat = str(df_tr['LAT'][i])
    #    key = lon+','+lat
    #    if key not in dic_loc_key.keys():
    #        dic_loc_key[key] = n
    #        n = n+1
    #
    #for i in range(len(df_te)):
    #    lon = str(df_te['LON'][i])
    #    lat = str(df_te['LAT'][i])
    #    key = lon+','+lat
    #    if key not in dic_loc_key.keys():
    #        dic_loc_key[key] = n
    #        n = n+1

    #f_key = open('../new/loction_key.txt_all','w')
    #f_key.write('LON,LAT,LOC_IDX\n')
    #for key ,locIdx in dic_loc_key.items():
    #    f_key.write(key+','+str(locIdx)+'\n')
    #f_key.close()

    #f_out = open('../new/hyk_mf_input.txt_all','w')
    #dic_locIdx_time = {}
    #for i in range(len(df_tr)):
    #    uid = str(int(df_tr['USERID'][i]))
    #    lon = str(df_tr['LON'][i])
    #    lat = str(df_tr['LAT'][i])
    #    loc_idx = dic_loc_key[lon+','+lat]
    #    key = uid+','+str(loc_idx)
    #    if key not in dic_locIdx_time.keys():
    #        dic_locIdx_time[key] = 1
    #    else:
    #        dic_locIdx_time[key] +=1

    #for i in range(len(df_te)):
    #    uid = str(int(df_te['USERID'][i]))
    #    lon = str(df_te['LON'][i])
    #    lat = str(df_te['LAT'][i])
    #    loc_idx = dic_loc_key[lon+','+lat]
    #    key = uid+','+str(loc_idx)
    #    if key not in dic_locIdx_time.keys():
    #        dic_locIdx_time[key] = 1
    #    else:
    #        dic_locIdx_time[key] +=1

    #for key,times in dic_locIdx_time.items():
    #    f_out.write(key+','+str(times)+'\n')


    ######################################################
    #### LDA input ######################################
    #f_in = open('../new/hyk_mf_input.txt_all','r')
    #dic = {}
    #for line in f_in:
    #    data = line.strip().split(',')
    #    uid = data[0]
    #    if uid not in dic.keys():
    #        loc_list = []
    #        loc_list.append(data[1])
    #        loc_list.append(data[2])
    #        dic[uid] = loc_list
    #    else:
    #        loc_list = dic[uid]
    #        loc_list.append(data[1])
    #        loc_list.append(data[2])
    #        dic[uid] = loc_list
    #f_out = open('../new/lda_input.txt_all','w')
    #for key,loc_list in dic.items():
    #    string = str(key)+":"
    #    for v in loc_list:
    #        string +=v+' '
    #    string = string[:-1]
    #    f_out.write(string+'\n')



    ############ mf uid-shopclass ##################
    df_tr = pd.read_csv('../new/TRAIN_NEW.csv',header=0)
    dic = {}
    for i in range(len(df_tr)):
        if df_tr['SHOPID'][i] ==0 :
            continue
        uid = int(df_tr['USERID'][i])
        shopclass = int(df_tr['SHOP_CLASS'][i])
        key = str(uid)+' '+str(shopclass)
        if key not in dic.keys():
            dic[key] = 1
        else:
            dic[key] += 1

    f_out = open('../new/mf_shoptype.txt','w')
    for k,v in dic.items():
        f_out.write(k+' '+str(v)+'\n')


#if __name__=='__main__':
#    df_tr = pd.read_csv(TRAIN_PATH,header=0)
#
#    dic = {}
#    for i in range(len(df_tr)):
#        uid = int(df_tr['USERID'][i])
#        shopid = int(df_tr['SHOPID'][i])
#        if shopid == 0:
#            continue
#        else:
#            key = str(uid)+" "+str(shopid)
#            if key not in dic.keys():
#                dic[key] = 1
#            else:
#                dic[key] += 1
#
#    f_out = open('mf_input.txt','w')
#    for key,v in dic.items():
#        f_out.write(key+" "+str(v)+'\n')
#    f_out.close()
