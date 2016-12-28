#!/bin/bash
# example running script for basicMF

# make buffer, transform text format to binary format
~/tools/svdfeature-1.2.2/tools/make_feature_buffer TRAIN_NEW_SVM.txt TRAIN_NEW_SVM.txt.buffer
~/tools/svdfeature-1.2.2/tools/make_feature_buffer TEST_NEW_SVM.txt TEST_NEW_SVM.txt.buffer

~/tools/svdfeature-1.2.2/svd_feature binaryClassification.conf num_round=500
~/tools/svdfeature-1.2.2/svd_feature_infer binaryClassification.conf pred=500


#../../tools/make_feature_buffer ua.base.example ua.base.buffer
#../../tools/make_feature_buffer ua.test.example ua.test.buffer
#
## training for 40 rounds
#../../svd_feature binaryClassification.conf num_round=40 
## write out prediction from 0040.model
#../../svd_feature_infer binaryClassification.conf pred=40 
