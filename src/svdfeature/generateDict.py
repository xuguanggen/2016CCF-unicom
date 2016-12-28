


if __name__=='__main__':
    uid_dict ={}
    shopid_dict = {}
    f_tr = open('../../new/TRAIN_SVMFORMAT.txt','r')
    f_te = open('../../new/TEST_SVMFORMAT.txt','r')

    f_tr_new = open('TRAIN_NEW_SVM.txt','w')
    f_te_new = open('TEST_NEW_SVM.txt','w')
    
    u_count = 0
    shop_count = 0

    for line in f_tr:
        data = line.strip().split(" ")
        uid = int(data[4].split(":")[0])
        shopid = int(data[-1].split(":")[0])

        if uid not in uid_dict.keys():
            uid_dict[uid]=u_count
            u_count +=1
        if shopid not in shopid_dict.keys():
            shopid_dict[shopid]=shop_count
            shop_count +=1
        string = data[0]+" "+data[1]+" "+data[2]+" "+data[3]+" "
        for i in range(4,len(data)-3):
            string += str(uid_dict[uid])+":"+data[i].split(":")[1]+" "
        for i in range(len(data)-3,len(data)):
            string += str(shopid_dict[shopid])+":"+data[i].split(":")[1]+" "
        string = string[:-1]+"\n"
        f_tr_new.write(string)

    for line in f_te:
        data = line.strip().split(" ")
        uid = int(data[4].split(":")[0])
        shopid = int(data[-1].split(":")[0])
        if uid not in uid_dict.keys():
            uid_dict[uid]=u_count
            u_count +=1
        if shopid not in shopid_dict.keys():
            shopid_dict[shopid]=shop_count
            shop_count +=1
        string = data[0]+" "+data[1]+" "+data[2]+" "+data[3]+" "
        for i in range(4,len(data)-3):
            string += str(uid_dict[uid])+":"+data[i].split(":")[1]+" "
        for i in range(len(data)-3,len(data)):
            string += str(shopid_dict[shopid])+":"+data[i].split(":")[1]+" "
        string = string[:-1]+"\n"
        f_te_new.write(string)


    f_key_uid = open('uid_key.txt','w')
    f_key_shopid = open('shopid_key.txt','w')
    for k,v in uid_dict.items():
        f_key_uid.write(str(k)+":"+str(v)+"\n")


    for k,v in shopid_dict.items():
        f_key_shopid.write(str(k)+":"+str(v)+"\n")
    
    f_tr.close()
    f_te.close()
