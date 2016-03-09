import argparse

import sys

import gc

import constants
from amr_parsing import write_parsed_amr
from model import Model
from parser import Parser
from preprocessing import preprocess
from smatch_v1_0.smatch_modified import *


def main(args):
    k = args.beam_size
    model_file_name = args.model
    test_file_name = args.test
    (precision, recall, f_score) = evaluate_model_by_file(model_file_name,test_file_name)
    print "Precision: %.3f" % precision
    print "Recall: %.3f" % recall
    print "Document F-score: %.3f" % f_score

def evaluate_model_by_file(model, gold_amr_file, k = 10, train = False):
    constants.FLAG_COREF = False
    constants.FLAG_PROP = False
    constants.FLAG_DEPPARSER = 'stdconv+charniak'

    test_instances = preprocess(gold_amr_file, START_SNLP=False, INPUT_AMR=True)
    p = Parser(model=model, oracle_type=constants.DET_T2G_ORACLE_ABT, action_type='basic', verbose=0, elog=sys.stdout)
    r = p.parse_beam_corpus_test(test_instances, k, Train=train)
    '''span_graph_pairs, results = parser.parse_corpus_test(test_instances, False)'''
    write_parsed_amr(r, test_instances, gold_amr_file, suffix='%s.parsed' % ('eval'))

    return eval(gold_amr_file + '.eval.parsed', gold_amr_file)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='hello')
    arg_parser.add_argument('-t','--test',default='./data/Little-Prince-amr-bank/test/test.txt',type=str,help='test amr file')
    arg_parser.add_argument('-b','--beam_size',default= 10,help='beam size')
    arg_parser.add_argument('-m','--model',default= './models/littlePrince-average.m',help='model')

    args = arg_parser.parse_args()
    if(True):
        model_file_names = []
        '''
        model_file_names.append('./models/littlePrince-beam-iter1.m')
        model_file_names.append('./models/littlePrince-beam-iter2.m')
        model_file_names.append('./models/littlePrince-beam-iter3.m')
        model_file_names.append('./models/littlePrince-beam-iter4.m')
        model_file_names.append('./models/littlePrince-beam-iter5.m')
        model_file_names.append('./models/littlePrince-beam-iter6.m')
        model_file_names.append('./models/littlePrince-beam-iter7.m')
        model_file_names.append('./models/littlePrince-beam-iter8.m')
        model_file_names.append('./models/littlePrince-beam-iter9.m')
        model_file_names.append('./models/littlePrince-beam-iter10.m')
        model_file_names.append('./models/littlePrince-beam-iter11.m')
        model_file_names.append('./models/littlePrince-beam-iter12.m')
        model_file_names.append('./models/littlePrince-beam-iter13.m')
        model_file_names.append('./models/littlePrince-beam-iter14.m')
        model_file_names.append('./models/littlePrince-beam-iter15.m')
        model_file_names.append('./models/littlePrince-beam-iter16.m')
        model_file_names.append('./models/littlePrince-beam-iter17.m')
        model_file_names.append('./models/littlePrince-beam-iter18.m')
        model_file_names.append('./models/littlePrince-beam-iter19.m')
        model_file_names.append('./models/littlePrince-beam-iter20.m')
        model_file_names.append('./models/littlePrince-beam-iter21.m')
        model_file_names.append('./models/littlePrince-beam-iter22.m')
        model_file_names.append('./models/littlePrince-beam-iter23.m')
        model_file_names.append('./models/littlePrince-beam-iter24.m')
        model_file_names.append('./models/littlePrince-beam-iter25.m')
        model_file_names.append('./models/littlePrince-beam-iter26.m')
        model_file_names.append('./models/littlePrince-beam-iter27.m')
        model_file_names.append('./models/littlePrince-beam-iter28.m')
        model_file_names.append('./models/littlePrince-beam-iter29.m')
        model_file_names.append('./models/littlePrince-beam-iter30.m')
        '''
        model_file_names.append('./models/littlePrince-average.m')

        results = []
        for model_file_name in model_file_names:
            model = Model.load_model(model_file_name)

            (precision, recall, f_score) = evaluate_model_by_file(model,'./data/Little-Prince-amr-bank/test/test.txt',5)
            print>>sys.stderr, "Model: " + model_file_name+" k: 5 averaged: True"
            print>>sys.stderr, "Precision: %.3f" % precision
            print>>sys.stderr, "Recall: %.3f" % recall
            print>>sys.stderr, "Document F-score: %.3f" % f_score
            s ="Model: " + model_file_name+" k: 5 averaged: True"
            s += " Precision: %.3f" % precision
            s += " Recall: %.3f" % recall
            s += " F-score: %.3f" % f_score
            results.append(s)

            '''
            (precision, recall, f_score) = evaluate_model_by_file(model,'./data/Little-Prince-amr-bank/test/test.txt',5, True)
            print>>sys.stderr, "Model: " + model_file_name+" k: 5 averaged: False"
            print>>sys.stderr, "Precision: %.3f" % precision
            print>>sys.stderr, "Recall: %.3f" % recall
            print>>sys.stderr, "Document F-score: %.3f" % f_score
            s ="Model: " + model_file_name+" k: 5 averaged: False"
            s += " Precision: %.3f" % precision
            s += " Recall: %.3f" % recall
            s += " F-score: %.3f" % f_score
            results.append(s)

            (precision, recall, f_score) = evaluate_model_by_file(model,'./data/Little-Prince-amr-bank/test/test.txt',10)
            print>>sys.stderr, "Model: " + model_file_name+" k: 10 averaged: True"
            print>>sys.stderr, "Precision: %.3f" % precision
            print>>sys.stderr, "Recall: %.3f" % recall
            print>>sys.stderr, "Document F-score: %.3f" % f_score
            s ="Model: " + model_file_name+" k: 10 averaged: True"
            s += " Precision: %.3f" % precision
            s += " Recall: %.3f" % recall
            s += " F-score: %.3f" % f_score
            results.append(s)


            (precision, recall, f_score) = evaluate_model_by_file(model,'./data/Little-Prince-amr-bank/test/test.txt',10, True)
            print>>sys.stderr, "Model: " + model_file_name+" k: 10 averaged: False"
            print>>sys.stderr, "Precision: %.3f" % precision
            print>>sys.stderr, "Recall: %.3f" % recall
            print>>sys.stderr, "Document F-score: %.3f" % f_score
            s ="Model: " + model_file_name+" k: 10 averaged: False"
            s += " Precision: %.3f" % precision
            s += " Recall: %.3f" % recall
            s += " F-score: %.3f" % f_score
            results.append(s)
            '''
            del model
            gc.collect()
        for result in  results:
            print >>sys.stderr, result
    else:
        main(args)