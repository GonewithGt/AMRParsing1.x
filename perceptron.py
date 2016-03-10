#!/usr/bin/python

import numpy as np
from constants import WEIGHT_DTYPE

class Perceptron():
    
    #model = None
    #num_updates = 0
    #wstep = 1

    def __init__(self,model):
        self.model = model
        self.num_updates = 0
        #self.reshape_rate = reshape_rate 

    def get_num_updates(self):
        return self.num_updates
                
    def no_update(self):
        self.model.wstep += 1

    def reshape_weight(self,act_idx,reshape_rate=10**5):        
        w = self.model.weight[act_idx]
        aw = self.model.aux_weight[act_idx]
        avgw = self.model.avg_weight[act_idx]

        #self.model.weight[act_idx] = np.vstack((w,np.zeros(shape=(reshape_rate,w.shape[1]),dtype=WEIGHT_DTYPE)))
        #self.model.aux_weight[act_idx] = np.vstack((aw,np.zeros(shape=(reshape_rate,aw.shape[1]),dtype=WEIGHT_DTYPE)))
        #self.model.avg_weight[act_idx] = np.vstack((avgw,np.zeros(shape=(reshape_rate,avgw.shape[1]),dtype=WEIGHT_DTYPE)))
        self.model.weight[act_idx] = np.concatenate((w,np.asarray([{}])))
        self.model.aux_weight[act_idx] = np.concatenate((w,np.asarray([{}])))
        self.model.avg_weight[act_idx] = np.concatenate((w,np.asarray([{}])))

    def update_weight_one_step(self,act_g,feat_g,act_l_g,act_b,feat_b,act_l_b):
        self.num_updates += 1
        
        act_g_idx = self.model.class_codebook.get_index(act_g)
        act_b_idx = self.model.class_codebook.get_index(act_b)

        act_l_g = act_l_g if act_l_g else 0
        #act_t_g = act_t_g if act_t_g else 0

        act_l_b = act_l_b if act_l_b else 0
        #act_t_b = act_t_b if act_t_b else 0
        
       ## if self.model.weight[act_g_idx].shape[0] <= self.model.feature_codebook[act_g_idx].size()+len(feat_g):
       ##     self.reshape_weight(act_g_idx)

        g_feats_indices = map(self.model.feature_codebook[act_g_idx].get_default_index,feat_g)
        for g_feat_ind in g_feats_indices:
            self.model.increase_weight_at(self.model.weight,act_g_idx,act_l_g, g_feat_ind,1.0)
            self.model.increase_weight_at(self.model.aux_weight,act_g_idx,act_l_g, g_feat_ind,float(self.model.wstep))

        b_feats_indices = map(self.model.feature_codebook[act_b_idx].get_default_index,feat_b)
        for b_feat_ind in b_feats_indices:
            self.model.increase_weight_at(self.model.weight,act_b_idx, act_l_b,b_feat_ind, -1.0)
            self.model.increase_weight_at(self.model.aux_weight,act_b_idx,act_l_b,b_feat_ind, -float(self.model.wstep))

        self.model.wstep += 1



    def average_weight(self):
        for i in self.model.class_codebook.indexes():
            weight = self.model.weight[i]
            aux_weight = self.model.aux_weight[i]
            avg_weight = self.model.avg_weight[i]
            wstep = self.model.wstep 
            
            #np.divide(aux_weight,wstep+.0,aux_weight)
            np.divide(aux_weight,wstep+.0,avg_weight)
            np.subtract(weight,avg_weight,avg_weight)
        
