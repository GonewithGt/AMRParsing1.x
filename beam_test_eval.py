import argparse

import sys

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
    (precision, recall, f_score) = evaluate_model_by_file(Model.load_model(model_file_name),test_file_name)
    print "Precision: %.3f" % precision
    print "Recall: %.3f" % recall
    print "Document F-score: %.3f" % f_score

def evaluate_model_by_file(model, gold_amr_file):
    constants.FLAG_COREF = False
    constants.FLAG_PROP = False
    constants.FLAG_DEPPARSER = 'stdconv+charniak'

    test_instances = preprocess(gold_amr_file, START_SNLP=False, INPUT_AMR=True)
    parser = Parser(model=model, oracle_type=constants.DET_T2G_ORACLE_ABT, action_type='basic', verbose=0, elog=sys.stdout)
    results = parser.parse_beam_corpus_test(test_instances, 5)
    '''span_graph_pairs, results = parser.parse_corpus_test(test_instances, False)'''
    write_parsed_amr(results, test_instances, gold_amr_file, suffix='%s.parsed' % ('eval'))
    return eval(gold_amr_file + '.eval.parsed', gold_amr_file)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='hello')
    arg_parser.add_argument('-t','--test',default='./data/Little-Prince-amr-bank/test/test.txt',type=str,help='test amr file')
    arg_parser.add_argument('-b','--beam_size',default= 10,help='beam size')
    arg_parser.add_argument('-m','--model',default= './models/littlePrince-average.m',help='model')

    args = arg_parser.parse_args()

    main(args)