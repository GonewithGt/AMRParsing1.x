#!/usr/bin/python

# transition-based (incremental) AMR parser
# author Chuan Wang
# March 28,2014
from __future__ import absolute_import
from common.util import *
from constants import *
from graphstate import GraphState
from newstate import Newstate
import optparse
import sys,copy,time,datetime
import numpy as np

from oracle import DetOracleABT, DetOracleSC, DetOracle
from perceptron import Perceptron
import cPickle as pickle
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import Queue

DRAW_GRAPH = False
WRITE_FAKE_AMR = False
OUTPUT_PARSED_AMR = True

class ScoredElement(object):
        def __init__(self, el, score):
            self.el = el
            self.score = score

        def __cmp__(self, other):
            return cmp(self.score, other.score)

        def __str__(self):
            return str(self.el) + ', ' + str(self.score)

class StackNode(object):
    def __init__(self,element, prev):
        self.element = element
        self.prev = prev


    def to_list(self):
        current = self
        result = []
        while not current.prev is None:
            result.append(current.element)
            current = current.prev
        result.append(current.element)
        result.reverse()
        return result

class BeamTrainState(object):
    def __init__(self,is_gold,act_ind, label_ind, act, label, features,score):
        self.is_gold = is_gold
        self.act_ind=act_ind
        self.label_ind =label_ind
        self.act = act
        self.label = label
        self.features = features
        self.score =score

class Parser(object):
    """
    """
    State = None
    #model = None
    oracle = None
    #action_table = None
    cm = None # confusion matrix for error analysis
    rtx = None # array for store the rumtime data
    rty = None #
    
    def __init__(self,model=None,oracle_type=DETERMINE_TREE_TO_GRAPH_ORACLE_SC,action_type='basic',verbose=1,elog=sys.stdout):
        self.sent = ''
        self.oracle_type=oracle_type
        self.verbose = verbose
        self.elog = elog
        self.model = model
        if self.oracle_type == DETERMINE_TREE_TO_GRAPH_ORACLE:
            Parser.State = __import__("graphstate").GraphState
            Parser.State.init_action_table(ACTION_TYPE_TABLE[action_type])
            Parser.oracle = __import__("oracle").DetOracle(self.verbose)
        elif self.oracle_type == DETERMINE_TREE_TO_GRAPH_ORACLE_SC:
            Parser.State = __import__("graphstate").GraphState
            Parser.State.init_action_table(ACTION_TYPE_TABLE[action_type])
            Parser.oracle = __import__("oracle").DetOracleSC(self.verbose)
        elif self.oracle_type == DET_T2G_ORACLE_ABT:
            Parser.State = __import__("graphstate").GraphState
            Parser.State.init_action_table(ACTION_TYPE_TABLE[action_type])
            Parser.oracle = __import__("oracle").DetOracleABT(self.verbose)
        elif self.oracle_type ==  DETERMINE_STRING_TO_GRAPH_ORACLE:
            Parser.State = __import__("newstate").Newstate
        else:
            pass
        self.perceptron = Perceptron(model)
        Parser.State.model = model


    def get_best_act(self,scores,actions):
        best_label_index = None
        best_act_ind = np.argmax(map(np.amax,scores))
        best_act = actions[best_act_ind]
        if best_act['type'] in ACTION_WITH_EDGE or best_act['type'] in ACTION_WITH_TAG:
            best_label_index = scores[best_act_ind].argmax()
        return best_act_ind, best_label_index

    def k_best_new(self,state, actions, features, k, train = True):
        beam = Queue.PriorityQueue()
        for i, (act, feat) in enumerate(zip(actions,features),0):

            act_ind  = self.model.class_codebook.get_index(act['type'])

            if (not act['type'] in ACTION_WITH_EDGE) and (not  act['type'] in ACTION_WITH_TAG):
                score = state.get_score_new(act_ind, 0, feat,train)
                if(beam.qsize()==k and score <= beam.queue[0].score):
                    continue
                beam.put(ScoredElement((i,act_ind, 0),score))
                if beam.qsize() > k:
                        beam.get()
            else:

                for label_ind in range(0, len(self.model.weight[act_ind])):
                    score = state.get_score_new(act_ind, label_ind, feat,train)
                    if(beam.qsize()==k and score <= beam.queue[0].score):
                        continue
                    #if act['type'] in ACTION_WITH_EDGE or act['type'] in ACTION_WITH_TAG:
                    if label_ind==1 and (not act['type'] == NEXT1):
                        continue
                    beam.put(ScoredElement((i,act_ind,label_ind),score))
                    #else:
                    #    beam.put(ScoredElement((i,act_ind,0),score))
                    if beam.qsize() > k:
                            beam.get()

        return self.priority_queue_to_list(beam)

    def get_k_best_act(self, scores,actions,k):
        beam = Queue.PriorityQueue()
        for act_ind in range(0, len(scores)):
            for label_ind in range(0, len(scores[act_ind])):
                act = actions[act_ind]
                if act['type'] in ACTION_WITH_EDGE or act['type'] in ACTION_WITH_TAG:
                    beam.put(ScoredElement((act_ind,label_ind),scores[act_ind][label_ind]))
                else:
                    beam.put(ScoredElement((act_ind,None),scores[act_ind][label_ind]))
                if beam.qsize() > k:
                    beam.get()
        return self.priority_queue_to_list(beam)


    def get_best_act_constraint(self,scores,actions,argset):
        best_label_index = None
        best_act_ind = np.argmax(map(np.amax,scores))
        if actions[best_act_ind]['type'] in ACTION_WITH_EDGE:
            best_label_index = scores[best_act_ind].argmax()
            # best label violates the constraint
            while best_label_index in argset:
                scores[best_act_ind][best_label_index] = -float('inf')
                best_act_ind = np.argmax(map(np.amax,scores))
                if actions[best_act_ind]['type'] in ACTION_WITH_EDGE or actions[best_act_ind]['type'] in ACTION_WITH_TAG:
                    best_label_index = scores[best_act_ind].argmax()
                else:
                    best_label_index = None
        elif actions[best_act_ind]['type'] in ACTION_WITH_TAG:
            best_label_index = scores[best_act_ind].argmax()
        return best_act_ind, best_label_index


    @staticmethod
    def get_label_index(act,label):
        if act['type'] in ACTION_WITH_EDGE:
            index = Parser.State.model.rel_codebook.get_index(label) if label is not None else 0
        elif act['type'] in ACTION_WITH_TAG:
            index = Parser.State.model.tag_codebook['ABTTag'].get_index(label)
        else:
            index = 0
        if label is None and act['type']== 0:
            index = 1
        return index

    @staticmethod
    def get_index_label(act,index):
        if act['type'] in ACTION_WITH_EDGE:
            label = Parser.State.model.rel_codebook.get_label(index) if index is not None else None
        elif act['type'] in ACTION_WITH_TAG:
            label = Parser.State.model.tag_codebook['ABTTag'].get_label(index)
        else:
            label = None
        #if(label == NULL_EDGE and (act['type'] == NEXT1) ):
        #   label = None
        return label

    def parse_corpus_train(self, instances, output_percent = 10):
        start_time = time.time()
        i =0
        output_shift = int(len(instances)/100) * output_percent
        for instance in instances:
            self.parse(instance)
            i = i + 1
            if i >= output_shift and i%(output_shift) ==0:
                print >> self.elog,"Training on %s instances takes %s" % (str(i),datetime.timedelta(seconds=round(time.time()-start_time,0)))
        print >> self.elog,"One pass on %s instances takes %s" % (str(len(instances)),datetime.timedelta(seconds=round(time.time()-start_time,0)))

    def parse_corpus_test(self, instances, output_percent = 10):
        start_time = time.time()
        parsed_amr = []
        i =0
        output_shift = int(len(instances)/100) * output_percent
        for inst in instances:
            i+=1
            _, state = self.parse(inst,False)
            parsed_amr.append(GraphState.get_parsed_amr(state.A))
            if i >= output_shift and i%(output_shift) ==0:
                print >> self.elog,"Parsing %s instances takes %s" % (str(i),datetime.timedelta(seconds=round(time.time()-start_time,0)))

        print >> self.elog,"One pass on %s instances takes %s" % (str(len(instances)),datetime.timedelta(seconds=round(time.time()-start_time,0)))
        return parsed_amr

    def contains_gold(self, beam):
        for scored_el in beam:
            state,is_gold = scored_el.el
            if is_gold:
                return True
        return False



    def need_to_terminate(self, beam):
        if len(beam) == 0:
            return True
        for scored_el in beam:
            (state,_), _ = scored_el.el
            if not state.is_terminal():
                return False
        return True

    def apply_oracle_acton(self, oracle_state, ref_graph, apply_features = True):

        violated = False
        gold_act, gold_label = Parser.oracle.give_ref_action(oracle_state,ref_graph)
        gold_actions = oracle_state.get_possible_actions(True)

        try:
            gold_act_ind = gold_actions.index(gold_act)
            gold_act_ind = self.model.class_codebook.get_index(gold_act['type'])
        except ValueError:
            if gold_act['type'] != NEXT2:
                violated = True # violated the constraint
            gold_actions.append(gold_act)
            gold_act_ind = len(gold_actions)-1

        gold_label_index = Parser.get_label_index(gold_act,gold_label)

        score = 0
        if apply_features:
            features = oracle_state.make_feat(gold_act)
            score = oracle_state.get_score_new(gold_act_ind,gold_label_index,features,True)

        if gold_act['type'] in ACTION_WITH_EDGE:
            gold_act['edge_label'] = gold_label
        elif gold_act['type'] in ACTION_WITH_TAG:
            gold_act['tag'] = gold_label

        return oracle_state.apply(gold_act), gold_act_ind,gold_label_index, gold_act, gold_label,violated,score

    def train_beam_max_violation(self, instance, k =10):
        ref_graph = instance.gold_graph
        oracle_conf = None
        max_conf = None
        beam = []
        max_dif = -1

        if True:
            oracle_state,gold_act_ind,gold_label_index, gold_act, gold_label,violated,score = self.apply_oracle_acton(Parser.State.init_state(instance,self.verbose),ref_graph,False)
            oracle_conf = ScoredElement(((oracle_state,True), StackNode((gold_act_ind,gold_label_index, gold_act, gold_label,violated, True), None)), score)
            beam = [ScoredElement(((oracle_state.pcopy(),True), StackNode((gold_act_ind,gold_label_index, gold_act, gold_label,violated,True), None)), score)]
            max_conf = beam[0]

        while not self.need_to_terminate(beam):
            gold_act_ind = gold_act = gold_label = gold_label_index = None
            if not oracle_conf.el[0][0].is_terminal():
                prev_score = oracle_conf.score
                oracle_state,gold_act_ind,gold_label_index, gold_act, gold_label,violated,score = self.apply_oracle_acton(oracle_conf.el[0][0],ref_graph,False)
                oracle_conf = ScoredElement(((oracle_state,True), StackNode((gold_act_ind,gold_label_index, gold_act, gold_label,violated, True), oracle_conf.el[1])), prev_score+score)

            old_beam = beam
            beam = Queue.PriorityQueue()
            for scored_state in old_beam:
                (state, is_gold), stack_node = scored_state.el

                if state.is_terminal():
                    beam.put(scored_state)
                    if beam.qsize() > k:
                        beam.get()
                    continue

                actions = state.get_possible_actions(True)

                features = map(state.make_feat,actions)

                for scored_el in self.k_best_new(state,actions, features,k):
                    (local_act_ind,act_ind,label_ind) = scored_el.el
                    act = actions[local_act_ind]
                    label = Parser.get_index_label(act,label_ind)
                    is_current_gold = False

                    if   is_gold and act_ind ==gold_act_ind and (label == gold_label):
                        is_current_gold = True
                    if act['type'] in ACTION_WITH_EDGE:
                        act['edge_label'] = label
                    elif act['type'] in ACTION_WITH_TAG:
                        act['tag'] = label
                    if is_gold and (not is_current_gold) and gold_act_ind==act_ind and label_ind==gold_label_index:
                        print 'type = '+ str(act['type']) +', gl = '+ ('None' if gold_label is None else gold_label)+', label = '+('None' if label is None else label)
                    new_state = state.pcopy().apply(act)
                    new_state_is_gold = is_current_gold
                    new_state_stack_node = StackNode((act_ind,label_ind,act,label, False, new_state_is_gold), stack_node)
                    new_scored_state = ScoredElement(((new_state,new_state_is_gold),new_state_stack_node),scored_state.score+ scored_el.score )

                    beam.put(new_scored_state)

                    if beam.qsize() > k:
                        beam.get()

            beam = self.priority_queue_to_list(beam)
            beam_len = len(beam)
            if beam_len > 0:
                highest_scored_conf = beam[beam_len - 1]
                (state, is_gold), _ = highest_scored_conf.el
                if (not is_gold) and ((highest_scored_conf.score - oracle_conf.score)> max_dif):
                    max_conf = highest_scored_conf
                    max_dif = highest_scored_conf.score - oracle_conf.score

        _, gold_stack_node = oracle_conf.el
        gold_sequence = gold_stack_node.to_list()
        _, max_viol_stack_node = max_conf.el
        max_viol_sequence = max_viol_stack_node.to_list()
        matched = 0
        for el in max_viol_sequence:
            _,_,_,_,_,is_gold = el
            if not is_gold:
                break
            else:
                matched += 1

        gold_state  = Parser.State.init_state(instance,self.verbose)
        max_viol_state  = Parser.State.init_state(instance,self.verbose)

        for el in gold_sequence[0:matched]:
            _,_,act,_, _,_ =el
            gold_state = gold_state.apply(act)

        for el in max_viol_sequence[0:matched]:
            _,_,act,_, _,_ =el
            max_viol_state = max_viol_state.apply(act)

        was_violated = False
        violated_indexes = []
        for i,el in enumerate(gold_sequence[matched:]):
            _,label_ind,act,_, violated,_ = el
            if not violated:
                feat = gold_state.make_feat(act)
                self.perceptron.part_update_weight(act['type'],feat, label_ind,1.0)
            else:
                violated_indexes.append(i)
                if not was_violated:
                       print "action "+ str(act['type'])+" seems to be illegal for " + str(ref_graph)
                was_violated = True
            gold_state = gold_state.apply(act)
        if  (not len(gold_state.action_history)== len(oracle_conf.el[0][0].action_history)) or ( (not gold_state.is_terminal()) == oracle_conf.el[0][0].is_terminal() ):
            print "shit 1"

        for i,el in enumerate(max_viol_sequence[matched:]):
            _,label_ind,act,_, _,_ = el
            feat = max_viol_state.make_feat(act)
            if max_viol_state.is_terminal() or not (act['type'] in [a['type'] for a in max_viol_state.get_possible_actions(True)]):
                print 'Wrong transition!'
            if len (violated_indexes) >0  and violated_indexes[0]==i:
                violated_indexes.pop()
            else:
                self.perceptron.part_update_weight(act['type'],feat, label_ind,-1.0)
            max_viol_state = max_viol_state.apply(act)
        if  (not len(max_viol_state.action_history)== len(max_conf.el[0][0].action_history)) or ( (not max_viol_state.is_terminal()) == max_conf.el[0][0].is_terminal() ):
            print "shit 2"

        self.perceptron.next_step()
        return len(gold_sequence)==len(max_viol_sequence) and len(gold_sequence)==matched


    def train_beam_multiple_oracles(self, instance, k = 10, oracles={DetOracleABT(),DetOracleSC(),DetOracle()}):

        def has_terminal_states(states):
            for _,state in states:
                if state.is_terminal():
                    return True
            return False
        #def get_golden_actions(oracle_states):
        #    for i,state in oracle_states:

        ref_graph = instance.gold_graph
        oracle_states = [(i,Parser.State.init_state(instance,self.verbose))for i in range(0,len(oracles))]

        beam = Queue.PriorityQueue()
        beam.put(ScoredElement((Parser.State.init_state(instance,self.verbose),True),0))
        step=0
        beam = self.priority_queue_to_list(beam)
        while (not has_terminal_states(oracle_states)) and len(beam) > 0:
            step += 1
            violated = False
            gold_act, gold_label = Parser.oracle.give_ref_action(oracle_state,ref_graph)
            gold_actions = oracle_state.get_possible_actions(True)
            try:
                gold_act_ind = gold_actions.index(gold_act)
                gold_act_ind = self.model.class_codebook.get_index(gold_act['type'])
            except ValueError:
                if gold_act['type'] != NEXT2:
                    violated = True # violated the constraint
                gold_actions.append(gold_act)
                gold_act_ind = len(gold_actions)-1
            if len(gold_actions) == 1 and step == 1:
                gold_features = None
            else:
                gold_features = oracle_state.make_feat(gold_act)

            gold_label_index = Parser.get_label_index(gold_act,gold_label)

            old_beam = beam
            beam = Queue.PriorityQueue()
            best_scored_state = None
            for scored_state in old_beam:
                state, is_gold = scored_state.el
                if(state.is_terminal()):
                    continue
                actions = state.get_possible_actions(True)
                if len(actions) == 1 and step ==1:
                    act = actions[0]
                    beam.put(ScoredElement((state.pcopy().apply(act),True),scored_state.score))
                    best_scored_state = BeamTrainState(True,0,0,act,None,None,0)
                else:
                    features = map(state.make_feat,actions)
                    #if(is_gold and gold_act_ind == len(actions)):
                    #    features.append(gold_features)
                    #    actions.append(gold_act)

                    #scores = map(state.get_score,(act['type'] for act in actions),features,[True]*len(actions))
                    for scored_el in self.k_best_new(state,actions, features,k):
                        (local_act_ind,act_ind,label_ind) = scored_el.el
                        act  = actions[local_act_ind]
                        act_label_score = scored_el.score
                        label = Parser.get_index_label(act,label_ind)
                        is_current_gold = False
                        if is_gold and act_ind ==gold_act_ind and (label == gold_label or gold_label_index==label_ind):
                            is_current_gold = True
                        if act['type'] in ACTION_WITH_EDGE:
                            act['edge_label'] = label
                        elif act['type'] in ACTION_WITH_TAG:
                            act['tag'] = label
                        new_scored_state = ScoredElement((state.pcopy().apply(act),is_current_gold),scored_state.score+act_label_score)
                        if best_scored_state is None or best_scored_state.score < new_scored_state.score:
                            best_scored_state = BeamTrainState(is_current_gold,act_ind,label_ind,act,label,features[local_act_ind],new_scored_state.score)
                        if beam.qsize()==k and best_scored_state.score<=beam.queue[0].score:
                            continue
                        beam.put(new_scored_state)
                        if beam.qsize() > k:
                            beam.get()
            beam = self.priority_queue_to_list(beam)
            if len(beam) == 0 or best_scored_state is None:
                return False

            if (gold_features is not None) and (not best_scored_state.is_gold) and (not violated) :
                self.perceptron.update_weight_one_step(gold_act['type'],gold_features,gold_label_index,best_scored_state.act['type'],best_scored_state.features,best_scored_state.label_ind)

            else:
                self.perceptron.no_update()
            best_act = gold_act
            best_label = gold_label
            act_to_apply = best_act
            if act_to_apply['type'] in ACTION_WITH_EDGE:
                act_to_apply['edge_label'] = best_label
            elif act_to_apply['type'] in ACTION_WITH_TAG:
                act_to_apply['tag'] = best_label

            oracle_state = oracle_state.apply(act_to_apply)
            if not self.contains_gold(beam):
                return False
                '''
                    for scored_state in old_beam:
                         state, is_gold = scored_state.el
                         if is_gold:
                            gold_act_score = state.get_score(gold_act['type'], gold_features, True)[gold_label_index]
                            beam.append(ScoredElement((oracle_state.pcopy(),True),scored_state.score+ gold_act_score))
                            break
                '''
        return True

    def train_beam_v_2(self, instance, k = 10):

        ref_graph = instance.gold_graph
        oracle_state = Parser.State.init_state(instance,self.verbose)

        beam = Queue.PriorityQueue()
        beam.put(ScoredElement((Parser.State.init_state(instance,self.verbose),True),0))
        step=0
        beam = self.priority_queue_to_list(beam)
        swapped = False
        while (not oracle_state.is_terminal()) and len(beam) > 0:
            step += 1
            violated = False
            gold_act, gold_label = Parser.oracle.give_ref_action(oracle_state,ref_graph)
            gold_actions = oracle_state.get_possible_actions(True)
            try:
                gold_act_ind = gold_actions.index(gold_act)
                gold_act_ind = self.model.class_codebook.get_index(gold_act['type'])
            except ValueError:
                if gold_act['type'] != NEXT2:
                    violated = True # violated the constraint
                gold_actions.append(gold_act)
                gold_act_ind = len(gold_actions)-1
            if len(gold_actions) == 1 and step == 1:
                gold_features = None
            else:
                gold_features = oracle_state.make_feat(gold_act)

            gold_label_index = Parser.get_label_index(gold_act,gold_label)

            old_beam = beam
            beam = Queue.PriorityQueue()
            best_scored_state = None
            for scored_state in old_beam:
                state, is_gold = scored_state.el
                if(state.is_terminal()):
                    continue
                actions = state.get_possible_actions(True)
                if len(actions) == 1 and step ==1:
                    act = actions[0]
                    beam.put(ScoredElement((state.pcopy().apply(act),True),scored_state.score))
                    best_scored_state = BeamTrainState(True,0,0,act,None,None,0)
                else:
                    features = map(state.make_feat,actions)
                    #if(is_gold and gold_act_ind == len(actions)):
                    #    features.append(gold_features)
                    #    actions.append(gold_act)

                    #scores = map(state.get_score,(act['type'] for act in actions),features,[True]*len(actions))
                    for scored_el in self.k_best_new(state,actions, features,k):
                        (local_act_ind,act_ind,label_ind) = scored_el.el
                        act  = actions[local_act_ind]
                        act_label_score = scored_el.score
                        label = Parser.get_index_label(act,label_ind)
                        is_current_gold = False
                        if is_gold and act_ind ==gold_act_ind and (label == gold_label or gold_label_index==label_ind):
                            is_current_gold = True
                        if act['type'] in ACTION_WITH_EDGE:
                            act['edge_label'] = label
                        elif act['type'] in ACTION_WITH_TAG:
                            act['tag'] = label
                        new_scored_state = ScoredElement((state.pcopy().apply(act),is_current_gold),scored_state.score+act_label_score)
                        if best_scored_state is None or best_scored_state.score < new_scored_state.score:
                            best_scored_state = BeamTrainState(is_current_gold,act_ind,label_ind,act,label,features[local_act_ind],new_scored_state.score)
                        if beam.qsize()==k and best_scored_state.score<=beam.queue[0].score:
                            continue
                        beam.put(new_scored_state)
                        if beam.qsize() > k:
                            beam.get()
            beam = self.priority_queue_to_list(beam)
            if len(beam) == 0 or best_scored_state is None:
                return False

            if (gold_features is not None) and (not best_scored_state.is_gold) and (not violated) :
                self.perceptron.update_weight_one_step(gold_act['type'],gold_features,gold_label_index,best_scored_state.act['type'],best_scored_state.features,best_scored_state.label_ind)

            else:
                self.perceptron.no_update()
            best_act = gold_act
            best_label = gold_label
            act_to_apply = best_act
            if act_to_apply['type'] in ACTION_WITH_EDGE:
                act_to_apply['edge_label'] = best_label
            elif act_to_apply['type'] in ACTION_WITH_TAG:
                act_to_apply['tag'] = best_label

            oracle_state = oracle_state.apply(act_to_apply)
            if not self.contains_gold(beam):
                swapped = True
                for scored_state in old_beam:
                     state, is_gold = scored_state.el
                     if is_gold:
                        gold_act_score = 0 if violated else state.get_score_new(gold_act_ind,gold_label_index,gold_features,True)
                        beam.append(ScoredElement((oracle_state.pcopy(),True),scored_state.score+ gold_act_score))
                        break
        return not swapped

    def train_beam(self, instance, k = 10):

        ref_graph = instance.gold_graph
        oracle_state = Parser.State.init_state(instance,self.verbose)

        beam = Queue.PriorityQueue()
        beam.put(ScoredElement((Parser.State.init_state(instance,self.verbose),True),0))
        step=0
        beam = self.priority_queue_to_list(beam)
        while (not oracle_state.is_terminal()) and len(beam) > 0:
            step += 1
            violated = False
            gold_act, gold_label = Parser.oracle.give_ref_action(oracle_state,ref_graph)
            gold_actions = oracle_state.get_possible_actions(True)
            try:
                gold_act_ind = gold_actions.index(gold_act)
                gold_act_ind = self.model.class_codebook.get_index(gold_act['type'])
            except ValueError:
                if gold_act['type'] != NEXT2:
                    violated = True # violated the constraint
                gold_actions.append(gold_act)
                gold_act_ind = len(gold_actions)-1
            if len(gold_actions) == 1 and step == 1:
                gold_features = None
            else:
                gold_features = oracle_state.make_feat(gold_act)

            gold_label_index = Parser.get_label_index(gold_act,gold_label)

            old_beam = beam
            beam = Queue.PriorityQueue()
            best_scored_state = None
            for scored_state in old_beam:
                state, is_gold = scored_state.el
                if(state.is_terminal()):
                    continue
                actions = state.get_possible_actions(True)
                if len(actions) == 1 and step ==1:
                    act = actions[0]
                    beam.put(ScoredElement((state.pcopy().apply(act),True),scored_state.score))
                    best_scored_state = BeamTrainState(True,0,0,act,None,None,0)
                else:
                    features = map(state.make_feat,actions)
                    #if(is_gold and gold_act_ind == len(actions)):
                    #    features.append(gold_features)
                    #    actions.append(gold_act)

                    #scores = map(state.get_score,(act['type'] for act in actions),features,[True]*len(actions))
                    for scored_el in self.k_best_new(state,actions, features,k):
                        (local_act_ind,act_ind,label_ind) = scored_el.el
                        act  = actions[local_act_ind]
                        act_label_score = scored_el.score
                        label = Parser.get_index_label(act,label_ind)
                        is_current_gold = False
                        if is_gold and act_ind ==gold_act_ind and (label == gold_label or gold_label_index==label_ind):
                            is_current_gold = True
                        if act['type'] in ACTION_WITH_EDGE:
                            act['edge_label'] = label
                        elif act['type'] in ACTION_WITH_TAG:
                            act['tag'] = label
                        new_scored_state = ScoredElement((state.pcopy().apply(act),is_current_gold),scored_state.score+act_label_score)
                        if best_scored_state is None or best_scored_state.score < new_scored_state.score:
                            best_scored_state = BeamTrainState(is_current_gold,act_ind,label_ind,act,label,features[local_act_ind],new_scored_state.score)
                        if beam.qsize()==k and best_scored_state.score<=beam.queue[0].score:
                            continue
                        beam.put(new_scored_state)
                        if beam.qsize() > k:
                            beam.get()
            beam = self.priority_queue_to_list(beam)
            if len(beam) == 0 or best_scored_state is None:
                return False

            if (gold_features is not None) and (not best_scored_state.is_gold) and (not violated) :
                self.perceptron.update_weight_one_step(gold_act['type'],gold_features,gold_label_index,best_scored_state.act['type'],best_scored_state.features,best_scored_state.label_ind)

            else:
                self.perceptron.no_update()
            best_act = gold_act
            best_label = gold_label
            act_to_apply = best_act
            if act_to_apply['type'] in ACTION_WITH_EDGE:
                act_to_apply['edge_label'] = best_label
            elif act_to_apply['type'] in ACTION_WITH_TAG:
                act_to_apply['tag'] = best_label

            oracle_state = oracle_state.apply(act_to_apply)
            if not self.contains_gold(beam):
                return False
                '''
                    for scored_state in old_beam:
                         state, is_gold = scored_state.el
                         if is_gold:
                            gold_act_score = state.get_score(gold_act['type'], gold_features, True)[gold_label_index]
                            beam.append(ScoredElement((oracle_state.pcopy(),True),scored_state.score+ gold_act_score))
                            break
                '''
        return True

    def parse_corpus_beam_train(self, instances, k, output_percent = 10):
        start_time = time.time()
        i =0
        parsed = 0
        output_shift = int(len(instances)/100) * output_percent
        for instance in instances:
            parsed += 1 if self.train_beam(instance,k) else 0
            i = i + 1
            if i >= output_shift and i%(output_shift) ==0:
                print >> self.elog,"Training on %s instances (parsed %s) takes %s" % (str(i),str(parsed),datetime.timedelta(seconds=round(time.time()-start_time,0)))
        print >> self.elog,"One pass on %s instances (parsed %s) takes %s" % (str(len(instances)),str(parsed),datetime.timedelta(seconds=round(time.time()-start_time,0)))

    def parse_corpus_beam_max_viol_train(self, instances, k, output_percent = 10):
        start_time = time.time()
        i =0
        parsed = 0
        output_shift = int(len(instances)/100) * output_percent
        for instance in instances:
            parsed += 1 if self.train_beam_max_violation(instance,k) else 0
            i = i + 1
            if i >= output_shift and i%(output_shift) ==0:
                print >> self.elog,"Training on %s instances (parsed %s) takes %s" % (str(i),str(parsed),datetime.timedelta(seconds=round(time.time()-start_time,0)))
        print >> self.elog,"One pass on %s instances (parsed %s) takes %s" % (str(len(instances)),str(parsed),datetime.timedelta(seconds=round(time.time()-start_time,0)))

    def parse_corpus_beam_train_v2(self, instances, k):
        start_time = time.time()
        i =0
        parsed = 0
        percent = int(len(instances)/100)
        for instance in instances:
            parsed += 1 if self.train_beam_v_2(instance,k) else 0
            i = i + 1
            if i >= percent and i%(percent) ==0:
                print >> self.elog,"Training on %s instances (parsed %s) takes %s" % (str(i),str(parsed),datetime.timedelta(seconds=round(time.time()-start_time,0)))
        print >> self.elog,"One pass on %s instances (parsed %s) takes %s" % (str(len(instances)),str(parsed),datetime.timedelta(seconds=round(time.time()-start_time,0)))

    def parse_beam_corpus_test(self, instances,k=10, skip = 0, Train = False, output_percent = 10):
        start_time = time.time()
        parsed_amr = []
        i =0
        skipped  = 0
        output_shift = int(len(instances)/100) * output_percent
        for inst in instances:
            if skipped< skip:
                skipped+=1
                continue
            i+=1
            state = self.parse_beam(inst,k, Train)
            parsed_amr.append(GraphState.get_parsed_amr(state.A))
            if i >= output_shift and i%(output_shift) ==0:
                print >> self.elog,"Parsing %s instances takes %s" % (str(i),datetime.timedelta(seconds=round(time.time()-start_time,0)))

        print >> self.elog,"One pass on %s instances takes %s" % (str(len(instances)),datetime.timedelta(seconds=round(time.time()-start_time,0)))
        return parsed_amr

    def _parse(self,instance):
        self.perceptron.no_update()
        return (True,Parser.State.init_state(instance,self.verbose))


    def priority_queue_to_list(self,queue):
        result =[]
        while not queue.empty():
            result.append(queue.get())
        return result

    def parse_beam(self, instance, k = 10, Train = False):
        beam = Queue.PriorityQueue()
        beam.put(ScoredElement(Parser.State.init_state(instance,self.verbose),0))
        best_parse = None
        step=0
        while not beam.empty():
            step += 1
            old_beam = self.priority_queue_to_list(beam)
            beam = Queue.PriorityQueue()
            for scored_state in old_beam:
                state = scored_state.el

                if state.is_terminal():
                    if best_parse is None or best_parse.score < scored_state.score:
                        best_parse = scored_state
                    continue

                actions = state.get_possible_actions(Train)
                if len(actions) == 1 :
                    act = actions[0]
                    beam.put(ScoredElement(state.pcopy().apply(act),scored_state.score))

                else:
                    features = map(state.make_feat,actions)
                    for scored_el in self.k_best_new(state,actions, features,k, Train):
                        (act_ind,unused,label_ind) = scored_el.el
                        act_label_score = scored_el.score
                        act = actions[act_ind]
                        label = Parser.get_index_label(act,label_ind)
                        if act['type'] in ACTION_WITH_EDGE:
                            act['edge_label'] = label
                        elif act['type'] in ACTION_WITH_TAG:
                            act['tag'] = label
                        beam.put(ScoredElement(state.pcopy().apply(act),scored_state.score+act_label_score))
                        if beam.qsize() > k:
                            beam.get()
        return best_parse.el

    def parse(self,instance,train=True): 
        # no beam; pseudo deterministic oracle
        state = Parser.State.init_state(instance,self.verbose)
        ref_graph = instance.gold_graph
        step = 0
        pre_state = None
        
        
        while not state.is_terminal():
            if self.verbose > 2:
                print >> sys.stderr, state.print_config()
            
            #start_time = time.time()
            violated = False
            actions = state.get_possible_actions(train)
            #argset = map(Parser.State.model.rel_codebook.get_index,list(state.get_current_argset()))
            #print "Done getactions, %s"%(round(time.time()-start_time,2))

            if len(actions) == 1:
                best_act = actions[0]
                best_label = None
            else:
                if train:
                    features = map(state.make_feat,actions)
                    #scores = map(state.get_score,(act['type'] for act in actions),features)

                    best_act_ind,_, best_label_index =self.k_best_new(state,actions,features,1)[0].el #self.get_best_act(scores,actions)#,argset)
                    best_act = actions[best_act_ind]
                    best_label = Parser.get_index_label(best_act,best_label_index)

                    #print "Done argmax, %s"%(round(time.time()-start_time,2))
                    #gold_act = getattr(self,self.oracle_type)(state,ref_graph)
                    gold_act, gold_label = Parser.oracle.give_ref_action(state,ref_graph)

                    try:
                        gold_act_ind = actions.index(gold_act)
                    except ValueError:
                        if self.verbose > 2:
                            print >> sys.stderr, 'WARNING: gold action %s not in possible action set %s'%(str(gold_act),str(actions))
                            
                        if gold_act['type'] != NEXT2:
                            violated = True # violated the constraint

                        actions.append(gold_act)
                        gold_act_ind = len(actions)-1
                        features.append(state.make_feat(gold_act))

                    gold_label_index = Parser.get_label_index(gold_act,gold_label)
                    '''
                    if gold_act['type'] in ACTION_WITH_EDGE:
                        gold_label_index = Parser.State.model.rel_codebook.get_index(gold_label)
                    elif gold_act['type'] in ACTION_WITH_TAG:
                        gold_label_index = Parser.State.model.tag_codebook['ABTTag'].get_index(gold_label)
                    else:
                        gold_label_index = None
                    '''

                    if self.verbose > 2:
                        print >> sys.stderr, "Step %s:take action %s gold action %s | State:sigma:%s beta:%s\n" % (step,actions[best_act_ind],actions[gold_act_ind],state.sigma,state.beta)

                    if (gold_act != best_act or gold_label != best_label) and not violated:
                        self.perceptron.update_weight_one_step(actions[gold_act_ind]['type'],features[gold_act_ind],gold_label_index,actions[best_act_ind]['type'],features[best_act_ind],best_label_index)
                        
                    else:
                        self.perceptron.no_update()

                    best_act = gold_act
                    best_label = gold_label
                    
                    #print "Done update, %s"%(round(time.time()-start_time,2))
                    #raw_input('ENTER TO CONTINUE')
                else:
                    features = map(state.make_feat,actions)
                    #scores = map(state.get_score,(act['type'] for act in actions),features)

                    best_act_ind,_, best_label_index = self.k_best_new(state,actions,features,1)[0].el #self.get_best_act(scores,actions)#,argset)

                    best_act = actions[best_act_ind]
                    best_label = Parser.get_index_label(best_act,best_label_index)

                    if self.verbose == 1:
                        gold_act, gold_label = Parser.oracle.give_ref_action(state,ref_graph)

                        self.evaluate_actions(actions[best_act_ind],best_label_index,gold_act,gold_label,ref_graph)
                    

                    if self.verbose > 2:
                        print >> sys.stderr, "Step %s: (%s,%s) | take action %s, label:%s | gold action %s,label:%s | State:sigma:%s beta:%s" % (step,actions[best_act_ind]['type'],gold_act['type'],actions[best_act_ind],best_label,gold_act,gold_label,state.sigma,state.beta)

                    if self.verbose > 3:
                        # correct next2 tag error
                        if gold_act['type'] == NEXT2:
                            self.output_weight(best_act_ind,best_label_index,features,actions)
                            if gold_act.get('tag',None) != best_act.get('tag',None) and 'tag' in gold_act and not (isinstance(gold_act['tag'],(ETag,ConstTag)) or re.match('\w+-\d+',gold_act['tag'])):
                                print >> sys.stderr, "Gold concept tag %s"%(gold_act['tag'])
                                
                            if gold_act in actions:                                
                                gold_act_ind = actions.index(gold_act)
                                gold_label_index = Parser.State.model.rel_codebook.get_index(gold_label)
                                self.output_weight(gold_act_ind,gold_label_index,features,actions)

                                
                    if self.verbose > 5:
                        # correct reentrance pair error
                        #if actions[best_act_ind]['type'] == REENTRANCE or gold_act['type'] == REENTRANCE:
                        self.output_weight(best_act_ind,best_label_index,features,actions)
                            #print >> sys.stderr, "incoming trace: %s" % (state.get_current_child().incoming_traces)
                            #if gold_act['type'] == REENTRANCE and gold_act['parent_to_add'] in [gov for rel,gov in state.get_current_child().incoming_traces]:
                            #    if gold_act not in actions:
                            #        import pdb
                            #        pdb.set_trace()
                            
                        if gold_act in actions:                                
                            gold_act_ind = actions.index(gold_act)
                            gold_label_index = Parser.State.model.rel_codebook.get_index(gold_label)
                            self.output_weight(gold_act_ind,gold_label_index,features,actions)

                        
            act_to_apply = best_act
            if act_to_apply['type'] in ACTION_WITH_EDGE:
                act_to_apply['edge_label'] = best_label
            elif act_to_apply['type'] in ACTION_WITH_TAG:
                act_to_apply['tag'] = best_label
            pre_state = state
            state = state.apply(act_to_apply)
            
            step += 1

        if self.verbose == 1:
            print >> sys.stderr, pre_state.print_config()

        return (step,state)

    def output_weight(self,act_ind,label_index,feats,actions):
        '''for debug '''
        label_ind = label_index if label_index is not None else 0
        feats_fired = feats[act_ind]
        act_idx = GraphState.model.class_codebook.get_index(actions[act_ind]['type']) 
        weight = GraphState.model.avg_weight[act_idx]
        feat_idx = map(GraphState.model.feature_codebook[act_idx].get_index,feats_fired)
        weight_sum = np.sum(weight[ [i for i in feat_idx if i is not None] ],axis = 0)
        #weight_fired = weight[[i for i in feat_idx if i is not None]]
        try:
            print >> sys.stderr, '\n'.join('%s,%f'%(f,weight[i][label_ind]) if i is not None else '%s,%f'%(f,0.0)  for f,i in zip(feats_fired,feat_idx))
            print >> sys.stderr, 'Sum: %f \n\n'%(weight_sum[label_ind])
        except TypeError:
            import pdb
            pdb.set_trace()
        #print >> sys.stderr,Parser.State.model.rel_codebook.get_label(0)
        
    def evaluate_actions(self,best_act,best_label_index,gold_act,gold_label,ref_graph):
        Parser.cm[gold_act['type'],best_act['type']] += 1.0

    def testUserGuide(self,instance):
        """take user input actions as guide"""
        state = Parser.State.init_state(instance,self.verbose)
        #for action in user_actions:
        while True:
            if state.is_terminal():
                return state

            print state.print_config()
            print state.A.print_tuples()
            action_str = raw_input('input action:')
            if not action_str:
                break                    
            act_type = int(action_str.split()[0])            
            if len(action_str) == 2:
                child_to_add = int(action_str.split()[1]) 
                action = {'type':act_type,'child_to_add':child_to_add}
            else:
                action = {'type':act_type}

            if state.is_permissible(action):
                state = state.apply(action)

            else:
                raise Error('Impermissibe action: %s'%(action))
            
        return state

    def draw_graph(self,fname,gtext):
        """ draw a graph using latex tikz/pgf """
        template = open("draw-graph/graph-template.tex",'r').read()
        fout = open("draw-graph/"+fname+".tex",'w')

        fout.write(template%(gtext))
        fout.close()
        
    def testOracleGuide(self,instance,start_step=0):
        """simulate the oracle's action sequence"""

        #if instance.comment['id'] == 'bolt12_10510_8841.3':
        #    self.verbose = 1
        #else:
        #    self.verbose = 0
        state = Parser.State.init_state(instance,self.verbose)
        ref_graph = state.gold_graph
        if state.A.is_root(): # empty dependency tree
            print >> sys.stderr, "Empty sentence! "+instance.text
            state.A = copy.deepcopy(ref_graph)
        step = 1
        if self.verbose > 1:
            #print "Gold graph:\n"+ref_graph.print_tuples()
            if DRAW_GRAPH:
                fname = "graph"+str(state.sentID)+"_gold"
                self.draw_graph(fname,ref_graph.getPGStyleGraph())

        while not state.is_terminal():

            if self.verbose > 0:
                print >> sys.stderr, state.print_config()
                #print state.A.print_tuples()                                    
                if DRAW_GRAPH:
                    fname = "graph"+str(state.sentID)+"_s"+str(step)
                    self.draw_graph(fname,state.A.getPGStyleGraph((state.idx,state.cidx)))
            

            if state.idx == START_ID:
                action,label = {'type':NEXT2},None
            else:
                action,label = Parser.oracle.give_ref_action(state,ref_graph)

            if self.verbose > 0:
                #print "Step %s:take action %s"%(step,action)
                print >> sys.stderr, "Step %s:take action %s, edge label %s | State:sigma:%s beta:%s" % (step,action,label,state.sigma,state.beta)
                '''
                print >> sys.stderr, [state.A.get_edge_label(state.idx,child) for child in state.A.nodes[state.idx].children if state.A.get_edge_label(state.idx,child).startswith('ARG') and child != state.cidx]
                if action['type'] in [REATTACH]:
                    node_to_add = action['parent_to_add'] if 'parent_to_add' in action else action['parent_to_attach']
                    path,_ = state.A.get_path(state.cidx,node_to_add)
                    path_str=[(state.sent[i]['pos'],state.sent[i]['rel']) for i in path[1:-1]]
                    path_str.insert(0,state.sent[path[0]]['rel'])
                    path_str.append(state.sent[path[-1]]['rel'])
                    print >> sys.stderr,'path for attachment', path, path_str #Parser.State.deptree.path(state.cidx),Parser.State.deptree.path(node_to_add),Parser.State.deptree.get_path(state.cidx,node_to_add)
                if action['type'] not in [NEXT2,DELETENODE]:
                    path,_ = GraphState.deptree.get_path(state.cidx,state.idx)
                    if state.A.nodes[state.idx].end - state.A.nodes[state.idx].start > 1:
                        path_pos_str = [(GraphState.sent[i]['pos'],GraphState.sent[i]['rel']) for i in path[1:-1] if i not in range(state.A.nodes[state.idx].start,state.A.nodes[state.idx].end)]
                    else:
                        path_pos_str = [(GraphState.sent[i]['pos'],GraphState.sent[i]['rel']) for i in path[1:-1]]
                    path_pos_str.insert(0,GraphState.sent[path[0]]['rel'])
                    path_pos_str.append(GraphState.sent[path[-1]]['rel'])
                    print >> sys.stderr,'path for current edge', path, path_pos_str
                    print >> sys.stderr,'Deleted children','b0',sorted([GraphState.sent[j]['form'].lower() for j in state.A.nodes[state.cidx].del_child]),'s0',sorted([GraphState.sent[j]['form'].lower() for j in state.A.nodes[state.idx].del_child])
                ''' 
            if state.is_permissible(action):
                if action['type'] in ACTION_WITH_EDGE:
                    action['edge_label'] = label
                elif action['type'] in ACTION_WITH_TAG:
                    action['tag'] = label
                else:
                    pass
                state = state.apply(action)
                step += 1
                #if self.verbose > 2 and step > start_step:
                #    raw_input('ENTER to continue')
            else:
                raise Error('Impermissibe action: %s'%(action))

        # deal with graph with no root
        state.A.post_process()
            
        return state

    def errorAnalyze(self,parsed_span_graph,gold_span_graph,instance,error_stat):
        """transformations as error types"""
            
        state = Parser.State.init_state(instance,self.verbose)
        seq = []
        for r in sorted(parsed_span_graph.multi_roots,reverse=True): seq += parsed_span_graph.postorder(root=r)
        seq.append(-1)
        sigma = Buffer(seq)        
        sigma.push(START_ID)
        state.sigma = sigma
        state.idx = sigma.top()
        state.A = parsed_span_graph
        ref_graph = gold_span_graph
        if state.A.is_root(): # empty dependency tree
            print >> sys.stderr, "Empty sentence! "+instance.text
            state.A = copy.deepcopy(ref_graph)
        step = 1


        while not state.is_terminal():

            if self.verbose > 0:
                print >> sys.stderr, state.print_config()            

            if state.idx == START_ID:
                action,label = {'type':NEXT2},None
            else:
                action,label = Parser.oracle.give_ref_action(state,ref_graph)

            if self.verbose > 0:
                #print "Step %s:take action %s"%(step,action)
                print >> sys.stderr, "Step %s:take action %s, edge label %s | State:sigma:%s beta:%s" % (step,action,label,state.sigma,state.beta)

            
            if state.is_permissible(action):
                if action['type'] == NEXT1:
                    if label != None and label != START_EDGE:
                        edge_label = state.A.get_edge_label(state.idx,state.cidx)
                        if edge_label != label:
                            error_stat['edge_error']['edge_label_error'][edge_label].append(state.sentID)
                elif action['type'] == NEXT2:
                    if label != None:
                        tag = state.get_current_node().tag
                        if tag != label:
                            error_stat['node_error']['node_tag_error'][tag].append(state.sentID)
                elif action['type'] == DELETENODE:
                    tag = state.get_current_node().tag
                    error_stat['node_error']['extra_node_error'][tag].append(state.sentID)
                elif action['type'] == INFER:                    
                    error_stat['node_error']['missing_node_error'][label].append(state.sentID)
                elif action['type'] in [REATTACH,REENTRANCE]:
                    btag = state.get_current_child().tag
                    bpos = GraphState.sent[state.cidx]['pos'] if isinstance(state.cidx,int) else btag
                    brel = GraphState.sent[state.cidx]['rel'] if isinstance(state.cidx,int) else btag
                    aid = action['parent_to_attach'] if action['type'] == REATTACH else action['parent_to_add']
                    atag = state.A.nodes[aid].tag
                    act_name = GraphState.action_table[action['type']]
                    if isinstance(aid,int):
                        apos = GraphState.sent[aid]['pos'] 
                        error_stat['edge_error'][act_name][bpos+brel+apos].append(state.sentID)
                    else:
                        apos = atag
                        error_stat['edge_error'][act_name][apos].append(state.sentID)
                else:
                    tag = state.get_current_node().tag
                    pos = GraphState.sent[state.idx]['pos'] if isinstance(state.idx,int) else tag
                    btag = state.A.nodes[state.cidx].tag
                    bpos = GraphState.sent[state.cidx]['pos'] if isinstance(state.cidx,int) else btag
                    act_name = GraphState.action_table[action['type']]
                    error_stat['edge_error'][act_name][pos+bpos].append(state.sentID)
                    
                
                if action['type'] in ACTION_WITH_EDGE:
                    action['edge_label'] = label
                elif action['type'] in ACTION_WITH_TAG:
                    action['tag'] = label
                else:
                    pass

                state = state.apply(action)
                step += 1
                #if self.verbose > 2 and step > start_step:
                #    raw_input('ENTER to continue')
            else:
                raise Error('Impermissibe action: %s'%(action))

        # deal with graph with no root
        state.A.post_process()
            
                                  
    def record_actions(self,outfile):
        output = open(outfile,'w')
        for act in list(Parser.State.new_actions):
            output.write(str(act)+'\n')
        output.close()


            
        
if __name__ == "__main__":
    pass
    

