#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shahab Sotudian
"""



import math
import numpy as np
import sklearn
import sys
import pandas as pd
import math
import six
from six.moves import range
from scipy.stats import rankdata
import time
from sklearn import preprocessing
from scipy.stats import spearmanr
import random
import copy
from numpy import asarray
from numpy import savetxt
import re

import lightgbm as lgb
from patsy import dmatrices
import pickle

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@                         Functions
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def Precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if len(r) != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def Average_Precision_at_k(r,K):
    r = np.asarray(r) != 0
    out = [Precision_at_k(r, i + 1) for i in range(K) if r[i]]
    if not out:
        return 0.
    return np.mean(out)

def mean_average_precision(rs):
    return np.mean([Average_Precision(r) for r in rs])


def Mean_Reciprocal_Rank(RGB):
    """.          
    RGB: Relevance scores (list or numpy) - first element is the first rank
    """
    Non_Zeros = (np.asarray(RGB).nonzero()[0])
    if Non_Zeros.size == 0:
        return 0.
    else:
        return 1. / (Non_Zeros[0] + 1)
    
    
    
def dcg_at_k(r, k):
    """
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider

    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):

    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


      

def iter_lines(lines, has_targets=True, one_indexed=True, missing=0.0):

    for line in lines:
                   
        data, _, comment = line.rstrip().partition('#')
        toks = data.split()

        num_features = 0
        x = np.repeat(missing, 8)
        y = -1.0
        if has_targets:
            # print(toks)
            # print("#########################")
            y = float(toks[0])
            toks = toks[1:]

        qid = _parse_qid_tok(toks[0])

        for tok in toks[1:]:
            fid, _, val = tok.partition(':')
            fid = int(fid)
            val = float(val)
            if one_indexed:
                fid -= 1
            assert fid >= 0
            while len(x) <= fid:
                orig = len(x)
                x.resize(len(x) * 2)
                x[orig:orig * 2] = missing

            x[fid] = val
            num_features = max(fid + 1, num_features)

        assert num_features > 0
        x.resize(num_features)

        yield (x, y, qid, comment)


   
def read_dataset(source, has_targets=True, one_indexed=True, missing=0.0):

    if isinstance(source, six.string_types):
        source = source.splitlines()

    max_width = 0
    xs, ys, qids, comments = [], [], [], []
    it = iter_lines(source, has_targets=has_targets,
                    one_indexed=one_indexed, missing=missing)
    for x, y, qid, comment in it:
        xs.append(x)
        ys.append(y)
        qids.append(qid)
        comments.append(comment)
        max_width = max(max_width, len(x))

    assert max_width > 0
    X = np.ndarray((len(xs), max_width), dtype=np.float64)
    X.fill(missing)
    for i, x in enumerate(xs):
        X[i, :len(x)] = x
    ys = np.array(ys) if has_targets else None
    qids = np.array(qids)
    comments = np.array(comments)

    return (X, ys, qids, comments)
   

def _parse_qid_tok(tok):
    assert tok.startswith('qid:')
    return tok[4:]

def TESTING_Func_LamdaRank(Validation_Unique_QID,Learned_Model_i,Unified_All_Y_QID_X):
   Validation_Num_Queries = len(Validation_Unique_QID)
   Predicted_NDCGatk = np.array([1, 5, 10, 25, 50, 100, 333333 ,1, 5, 10, 25, 50, 100])       
   NDCG_all = [0]*len(Predicted_NDCGatk)   
   for tt in range(Validation_Num_Queries):
       Index_ID=np.where(Unified_All_Y_QID_X[:,1] == Validation_Unique_QID[tt])[0]
       #dValid_CoMPlex = lgb.Dataset( Unified_All_Y_QID_X[Index_ID,2:] , label= np.expand_dims(Unified_All_Y_QID_X[Index_ID,0], axis=1)   )  
       #dValid_CoMPlex.set_group(group=np.array([M]).flatten()) # Grouping
       #Pred_Q = pd.DataFrame(Learned_Model_i.predict(dValid_CoMPlex))
       Pred_Q = pd.DataFrame(Learned_Model_i.predict(Unified_All_Y_QID_X[Index_ID,2:]))
       Ground_Q= Unified_All_Y_QID_X[Index_ID,0]
       Pred_Ranks=(rankdata(-Pred_Q, method='ordinal'))    # Decreasing order
       Concat = np.hstack([ np.expand_dims(Ground_Q, axis=1)  ,  np.expand_dims(Pred_Ranks, axis=1)  ])
       sorted_array = Concat[np.argsort(Concat[:, 1])]
       RGT= sorted_array[:,0]
       # Performance Metrics
       Set_NDCG = [ndcg_at_k(RGT, i) for i in [1, 5, min(10,len(RGT)), min(25,len(RGT)), min(50,len(RGT)), min(100,len(RGT))]]
       Set_Mean_Reciprocal_Rank = [Mean_Reciprocal_Rank(RGT)]
       Set_Average_Precision = [Average_Precision_at_k(RGT, i) for i in [1, 5, min(10,len(RGT)), min(25,len(RGT)), min(50,len(RGT)), min(100,len(RGT))]]
       All_Metrics = Set_NDCG + Set_Mean_Reciprocal_Rank + Set_Average_Precision
       NDCG_all = np.add(NDCG_all, All_Metrics)  
          
   Predicted_NDCG=NDCG_all/Validation_Num_Queries
   #Predicted_NDCGatk=np.vstack([Predicted_NDCGatk,Predicted_NDCG])
   return Predicted_NDCG 

def train_validate_test_split(All_QIDs, train_percent=0.60, validate_percent=0.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(All_QIDs)
    #print(perm)
    m = len(perm)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = perm[:train_end]
    validate = perm[train_end:validate_end]
    test = perm[validate_end:]
    return train, validate, test   



def LamdaRank_5FCV_Func(fold,Etas,Max_depths,Num_Trees,Col_sample_rates,Num_leaves,Type_Data,Splitting_Type,Num_Drugs):
  

    # Load data
    CellDrug_Blood = pd.read_csv('/home/BigData_CellDrug'+Type_Data+'.csv')
    CellDrug_Blood=CellDrug_Blood.drop(columns=['Index']) 
    
    CellGene_Blood = pd.read_csv('/home/BigData_CellGene'+Type_Data+'.csv')
    CellGene_Blood=CellGene_Blood.drop(columns=['Index']) 
    
    
    Num_Genes=CellGene_Blood.shape[1]    


    X=CellGene_Blood.iloc[:,0:Num_Genes]
    Y=CellDrug_Blood.iloc[:,0:Num_Drugs]
    
    Max_r =10
    All_QIDs = list(range(1,len(CellGene_Blood)+1))  
    
    if Type_Data == '_Blood_normilized':
        if Splitting_Type == 'Fixed':
            Tqids = list(range(1,52))
            Testqids = list(range(52,len(CellGene_Blood)+1))
            Vqids = Testqids
        elif Splitting_Type == 'Random_60_20_20':
            Tqids,Vqids,Testqids = train_validate_test_split(All_QIDs, train_percent=0.60, validate_percent=0.2, seed=fold)
            
    elif Type_Data == '_Immune_normilized':
        if Splitting_Type == 'Fixed':
            Tqids = list(range(1,49))
            Testqids = list(range(49,len(CellGene_Blood)+1))
            Vqids = Testqids
        elif Splitting_Type == 'Random_60_20_20':
            Tqids,Vqids,Testqids = train_validate_test_split(All_QIDs, train_percent=0.60, validate_percent=0.2, seed=fold)
    elif Type_Data == '_Lung_normilized':   
        if Splitting_Type == 'Fixed':
            Tqids = list(range(1,42))
            Testqids = list(range(42,len(CellGene_Blood)+1))
            Vqids = Testqids  
        elif Splitting_Type == 'Random_60_20_20':
            Tqids,Vqids,Testqids = train_validate_test_split(All_QIDs, train_percent=0.60, validate_percent=0.2, seed=fold)
    elif Type_Data == '':   
        if Splitting_Type == 'Fixed':
            Tqids = list(range(1,220))
            Vqids = list(range(220,280))            
            Testqids = list(range(280,len(CellGene_Blood)+1))
        elif Splitting_Type == 'Random_60_20_20':
            Tqids,Vqids,Testqids = train_validate_test_split(All_QIDs, train_percent=0.60, validate_percent=0.2, seed=fold)
            
        
    Final_X=[]
    Final_Y=[]
    
    # Training Data
    # Data generation - Create one-hot for all drugs
    for q in range(X.shape[0]):
        S_in=X.iloc[q,:].values
        S_out=list(Y.iloc[q,:])
        #S_out_Rel = [(1-S_out[j])*10 for j in range(len(S_out))] # convert to score since smaller better        
        S_out_Rel = np.round([(1-S_out[j])*10 for j in range(len(S_out))]) # convert to score since smaller better  # Round the labels
    
        OneHOT = []
        for rr in range(Num_Drugs):
            letter = [0]*Num_Drugs
            letter[rr] = 1
            OneHOT.append([[q+1]+letter + list(S_in)][0])
       
        Big_X_Relevance = np.expand_dims( np.array(S_out_Rel) , axis=1)
        #Multivariate_Y = Data_Prepration_Multi_Reg_NEWVersion_OutPULength(Big_X_Relevance, OutPut_Length,Alpha,Beta,Max_r,ScoreFunc,F1_Type,F2_Type)
     
        if q == 0:
            Final_X = np.array(OneHOT)
            #Final_Y = Multivariate_Y
            Final_Relevance = np.expand_dims( np.array(S_out_Rel) , axis=1)
        else:
            Final_X = np.vstack([Final_X,np.array(OneHOT)])
            #Final_Y = np.vstack([Final_Y,Multivariate_Y])
            Final_Relevance = np.vstack([Final_Relevance,np.expand_dims( np.array(S_out_Rel) , axis=1)])  
        
    Unified_All_Y_QID_X = np.hstack([Final_Relevance,Final_X])

    # =========-------------------------------------------------------------------
    #  Data
    # =========-------------------------------------------------------------------    

    # Train Data
    Train_Unique_QID = np.unique(Tqids).astype(np.float64)
    Group_List = []
    Cc =0 
    for qid in Train_Unique_QID:
         Index_ID=np.where(Unified_All_Y_QID_X[:,1] == qid)[0]  
         Big_X_features=Unified_All_Y_QID_X[Index_ID,2:]  # First column is IDs
         Big_X_Relevance= np.expand_dims(Unified_All_Y_QID_X[Index_ID,0], axis=1) 
         Group_List.append(len(Unified_All_Y_QID_X[Unified_All_Y_QID_X[:,1] == qid,:]))
         if Cc == 0:
             X_Train = Big_X_features
             Y_Train = Big_X_Relevance
         else:
             X_Train = np.vstack([X_Train,Big_X_features])
             Y_Train = np.vstack([Y_Train,Big_X_Relevance])         
         Cc +=1
        
        
    X_groups=np.array(Group_List).flatten()
    del Cc,Big_X_features,Big_X_Relevance,Index_ID,Group_List

    
    # Validation Data
    Validation_Unique_QID = np.unique(Vqids).astype(np.float64)
    Big_X_Valid = Unified_All_Y_QID_X[np.isin(Unified_All_Y_QID_X[:,1],Validation_Unique_QID),:]
    X_Valid = Big_X_Valid[:,2:]
    Y_Valid = Big_X_Valid[:,0]
    QID_Valid = Big_X_Valid[:,1]
    
    # Test Data
    Test_Unique_QID = np.unique(Testqids).astype(np.float64)
    Big_X_Test = Unified_All_Y_QID_X[np.isin(Unified_All_Y_QID_X[:,1],Test_Unique_QID),:]
    X_Test = Big_X_Test[:,2:]
    Y_Test = Big_X_Test[:,0]
    QID_Test = Big_X_Test[:,1]    
    
    
    # data lgb
    dtrain = lgb.Dataset(X_Train, label=Y_Train.flatten())
    dtest = lgb.Dataset(X_Test, label=Y_Test)
    
    # Grouping
    
    dtrain.set_group(group=X_groups) # Set the query_id values to DMatrix data structure
    

    

    
    

    # =========-------------------------------------------------------------------
    #  Training
    # =========-------------------------------------------------------------------    

    All_parameters = [] 
    All_Thetas = {}
    Counter = 0
    for MDL in Min_data_in_leaf_All:
        for eta in Etas:
            for MSHL in Min_sum_hessian_in_leaf_all:
                for NT in Num_Trees:
                    for NL in Num_leaves:    
                        params = {
                            "task": "train",
                             'boosting_type':'gbdt',
                             "num_leaves": NL,
                             'min_sum_hessian_in_leaf':MSHL,
                             'learning_rate':eta,
                             'n_estimators':NT,
                             'objective':'lambdarank',  # lambdarank rank_xendcg
                             'min_data_in_leaf':MDL,
                             'random_state':12345,
                             #"metric": "ndcg",
                             "num_threads": 4,
                             #'device_type':'gpu',
                             'verbosity':-1,
                             }
                        Learned_Model = lgb.train(params, dtrain)
                        
                        All_Thetas[Counter] = Learned_Model
                        Counter += 1
                        All_parameters.append([Counter, MDL, eta,MSHL,NT,NL])

    # Save Theta
    Name_save1 = 'All_Models_LamdaMart'+'_Fold_'+str(fold)+'.pkl'
    f = open(Name_save1,"wb")
    pickle.dump(All_Thetas,f)
    f.close()
    

    # =========-------------------------------------------------------------------
    #  Validation
    # =========-------------------------------------------------------------------    

    Validation_Unique_QID = np.unique(Vqids).astype(np.float64)    # Unique Queriy IDs + COnvert QIDs to float
    for pp in range(len(All_Thetas)):    
        Predicted_NDCG = TESTING_Func_LamdaRank(Validation_Unique_QID,All_Thetas[pp],Unified_All_Y_QID_X)
        if pp==0:
            arr = Predicted_NDCG
        else:
            arr = np.vstack((arr,Predicted_NDCG))
           
    Predicted_NDCGatk_Valid = pd.DataFrame(arr, columns=['NDCG@1', 'NDCG@5', 'NDCG@10', 'NDCG@25', 'NDCG@50', 'NDCG@100',
                                                    'Mean_Reciprocal_Rank' ,
                                                    'MAP@1', 'MAP@5', 'MAP@10', 'MAP@25', 'MAP@50', 'MAP@100'])
    
    del pp , Predicted_NDCG,arr
        
    
    
    # =========-------------------------------------------------------------------  
    # Best result parameters
    
    Top5_Top10 = Predicted_NDCGatk_Valid['NDCG@5'] + Predicted_NDCGatk_Valid['NDCG@10']   # Baset that have highest Top5 and Top10 NDCG performance
    Best_Itr = (np.where( Top5_Top10 == np.max(Top5_Top10))[0][0])  
    
    # =========-------------------------------------------------------------------
    #  Testing
    # =========-------------------------------------------------------------------  
    
    Test_Unique_QID = np.unique(Testqids).astype(np.float64)    # Unique Queriy IDs + COnvert QIDs to float
    Predicted_NDCG_t = TESTING_Func_LamdaRank(Test_Unique_QID,All_Thetas[Best_Itr],Unified_All_Y_QID_X)
 
    Predicted_NDCGatk_Test = pd.DataFrame(np.expand_dims(Predicted_NDCG_t, axis=0), columns=['NDCG@1', 'NDCG@5', 'NDCG@10', 'NDCG@25', 'NDCG@50', 'NDCG@100',
                                                    'Mean_Reciprocal_Rank' ,
                                                    'MAP@1', 'MAP@5', 'MAP@10', 'MAP@25', 'MAP@50', 'MAP@100'])
    
    print('\n ******     LamdaMART Test Performance     ******  \n',Predicted_NDCGatk_Test.T)
    print('\n ******     Best Parameters     ******  \n')
    print('Best min_data_in_leaf: ',All_parameters[Best_Itr][1],'\n')
    print('Best Etas: ',All_parameters[Best_Itr][2],'\n')
    print('Best min_sum_hessian_in_leaf: ',All_parameters[Best_Itr][3],'\n')
    print('Best Num_Trees: ',All_parameters[Best_Itr][4],'\n')
    print('Best num_leaves: ',All_parameters[Best_Itr][5],'\n')

    del Predicted_NDCG_t

    # =========-------------------------------------------------------------------
    #  Save Best Theta
    # =========-------------------------------------------------------------------    
    Name_save2 = 'Best_Model_LamdaMart'+'_Fold_'+str(fold)+'.pkl'
    f1 = open(Name_save2,"wb")
    Best_Model = All_Thetas[Best_Itr]
    pickle.dump(Best_Model,f1)
    f1.close()
    return Predicted_NDCGatk_Valid,Predicted_NDCGatk_Test,All_parameters[Best_Itr]




# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@                         5 FCV 
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# =========-------------------------------------------------------------------
#  Parameters
Num_Trees = [10,100,1000]
Etas = [1E-1,1E-2,1E-3]
Num_leaves = [10,100,200]  

Min_sum_hessian_in_leaf_all = [10,50,100] 
Min_data_in_leaf_All = [10,50,100] 





Type_Data = '_Blood_normilized'  #  '_Blood_normilized'   '_Immune_normilized'  '_Lung_normilized'  ''
Splitting_Type = 'Random_60_20_20'  # 'Random_60_20_20'  'Fixed'
Num_Drugs=50  # 179




Bet_Para_all = []
for fold in range(1,6):
    print('***********************************************')
    print('**** FOLD', fold, '  -----------------------')
    print('***********************************************') 
    Predicted_NDCGatk_Valid,Predicted_NDCGatk_Test,Best_Para = LamdaRank_5FCV_Func(fold,Etas,Min_sum_hessian_in_leaf_all,Num_Trees,Min_data_in_leaf_All,Num_leaves,Type_Data,Splitting_Type,Num_Drugs)
    A = ['Fold'+str(fold)]
    Bet_Para_all.append(Best_Para)
    Fold_Row = pd.DataFrame({'NDCG@1':A, 'NDCG@5':A, 'NDCG@10':A,'NDCG@25':A, 'NDCG@50':A, 'NDCG@100':A,
                       'Mean_Reciprocal_Rank':A ,'MAP@1':A, 'MAP@5':A,'MAP@10':A, 'MAP@25':A, 'MAP@50':A, 'MAP@100':A}, index =[0])
    
    df_Valid = pd.concat([Fold_Row, Predicted_NDCGatk_Valid]).reset_index(drop = True)
    df_Test = pd.concat([Fold_Row, Predicted_NDCGatk_Test]).reset_index(drop = True)
    if fold == 1:
        Performnace_Validation = df_Valid
        Performnace_Test = df_Test
    else:
        Performnace_Validation = Performnace_Validation.append(df_Valid , ignore_index=True)
        Performnace_Test = Performnace_Test.append(df_Test , ignore_index=True)
    del A,fold,Fold_Row, df_Valid,df_Test,Predicted_NDCGatk_Valid,Predicted_NDCGatk_Test
    

Performnace_Test_Average = np.zeros((1,Performnace_Test.shape[1]))
for q in [1,3,5,7,9]: Performnace_Test_Average = Performnace_Test_Average + 0.2*np.array(list(Performnace_Test.loc[q][:].values))
   