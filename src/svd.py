#! /usr/bin/env python

import numpy as np
from scipy.sparse.linalg import svds
from scipy import sparse


def vector_to_diagonal(vector):
    if (isinstance(vector, np.ndarray) and vector.ndim == 1) or \
            isinstance(vector, list):
        length = len(vector)
        diag_matrix = np.zeros((length, length))
        np.fill_diagonal(diag_matrix, vector)
        return diag_matrix
    return None



if __name__=='__main__':
    f_in = open('../new/hyk_mf_input.txt_all','r')
    
    dic = {}
    loc_set = set()
    for line in f_in:
        data = line.strip().split(',')
        uid = data[0]
        locid = data[1]
        times = data[2]
        loc_set.add(locid)
        if uid not in dic.keys():
            cur_list = []
            cur_list.append(locid)
            cur_list.append(times)
            dic[uid] = cur_list
        else:
            cur_list = dic[uid]
            cur_list.append(locid)
            cur_list.append(times)
            dic[uid] = cur_list

    uid_list = []
    mat = np.zeros((len(dic),len(loc_set)))
    i = 0
    for uid,cur_list in dic.items():
        uid_list.append(uid)
        for j in range(len(cur_list)/2):
            loc_id = int(cur_list[j*2])
            times  = int(cur_list[j*2+1])
            mat[i][loc_id] = times
        i += 1

    mat = mat.astype('float')
    U,S,V = svds(sparse.csr_matrix(mat),k=15,maxiter=200)

    i = 0
    f_out = open('../new/user_svd.txt','w')
    f_out.write('USERID,USER_SVDFEATURE\n')
    for uid in uid_list:
        string = uid+',"['
        for j in range(len(U[i])):
            string += str(U[i][j])+','
        string = string[:-1]+']"'
        f_out.write(string+'\n')
        i = i+1

    f_out.close()
    f_out = open('../new/loc_svd.txt','w')
    f_out.write('LOC_IDX,LOC_SVDFEATURE\n')
    V = np.transpose(V)
    for i in range(V.shape[0]):
        string = str(i)+',"['
        for j in range(len(V[i])):
            string += str(V[i][j])+','
        string = string[:-1]+']"'
        f_out.write(string+'\n')
