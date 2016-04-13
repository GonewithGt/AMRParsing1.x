#!/usr/bin/python

import numpy as np
from constants import WEIGHT_DTYPE

class Perceptron():

    def __init__(self,model):
        self.model = model
        self.num_updates = 0

    def get_num_updates(self):
        return self.num_updates
                
    def no_update(self):
        self.model.wstep += 1


    def update_weight_one_step(self,act_g,feat_g,act_l_g,act_b,feat_b,act_l_b):
        self.num_updates += 1
        
        act_g_idx = self.model.class_codebook.get_index(act_g)
        act_b_idx = self.model.class_codebook.get_index(act_b)

        act_l_g = act_l_g if act_l_g else 0

        act_l_b = act_l_b if act_l_b else 0

        g_feats_indices = map(self.model.feature_codebook[act_g_idx].get_default_index,feat_g)
        for g_feat_ind in g_feats_indices:
            self.model.increase_weight_at(self.model.weight,act_g_idx,act_l_g, g_feat_ind,1.0)
            self.model.increase_weight_at(self.model.aux_weight,act_g_idx,act_l_g, g_feat_ind,float(self.model.wstep))

        b_feats_indices = map(self.model.feature_codebook[act_b_idx].get_default_index,feat_b)
        for b_feat_ind in b_feats_indices:
            self.model.increase_weight_at(self.model.weight,act_b_idx, act_l_b,b_feat_ind, -1.0)
            self.model.increase_weight_at(self.model.aux_weight,act_b_idx,act_l_b,b_feat_ind, -float(self.model.wstep))

        self.model.wstep += 1


    def part_update_weight(self,act,feat,label_index, delta):


        act_index = self.model.class_codebook.get_index(act)

        label_index = label_index if label_index else 0

        feat_indices = map(self.model.feature_codebook[act_index].get_default_index,feat)
        for feat_index in feat_indices:
            self.model.increase_weight_at(self.model.weight,act_index,label_index, feat_index,delta)
            self.model.increase_weight_at(self.model.aux_weight,act_index,label_index, feat_index,delta *float(self.model.wstep))

    def next_step(self):

        self.num_updates += 1
        self.model.wstep += 1




    def average_weight(self):
        for act_ind in self.model.class_codebook.indexes():
            weight = self.model.weight[act_ind]
            aux_weight = self.model.aux_weight[act_ind]
            avg_weight = self.model.avg_weight[act_ind]
            wstep = self.model.wstep 

            for label_ind in range(0,len(weight)):
                for feat_ind in set(weight[label_ind].keys()).union(set(aux_weight[label_ind].keys())).union(set(avg_weight[label_ind].keys())):

                    v = self.model.weight_at(self.model.weight, act_ind,label_ind,feat_ind)-self.model.weight_at(self.model.aux_weight, act_ind,label_ind,feat_ind)/float(wstep)
                    self.model.update_weight(self.model.avg_weight,act_ind,label_ind,feat_ind,v)

            #np.divide(aux_weight,wstep+.0,aux_weight)
            #np.divide(aux_weight,wstep+.0,avg_weight)
            #np.subtract(weight,avg_weight,avg_weight)
        
