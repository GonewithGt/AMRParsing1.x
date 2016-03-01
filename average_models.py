import argparse
import os

import gc

import constants
from constants import DET_T2G_ORACLE_ABT
from parser import Parser
from smatch_v1_0.smatch_modified import *
from model import Model
from amr_parsing import preprocess, write_parsed_amr
import Queue as Q

reload(sys)
sys.setdefaultencoding('utf-8')


def main(args):
    basic_model_name =args.basic
    dev_amr_file = args.dev
    gold_amr_file = args.test
    k = args.k
    '''
    average(['./models/littlePrince-iter1.m', './models/littlePrince-iter25.m'], './models/littlePrince-avg.m')
    evaluate('./data/Little-Prince-amr-bank/test/test.txt.all.parsed', './data/Little-Prince-amr-bank/test/test.txt')

    if (True):
        test_queue()
        return
    model_names = []
    model_names.append('./models/littlePrince-iter4.m')
    model_names.append('./models/littlePrince-iter5.m')
    model_names.append('./models/littlePrince-iter6.m')
    model_names.append('./models/littlePrince-iter3.m')
    average(model_names, './models/littlePrince-average.m')
    print "evaluating ./models/littlePrince-average.m"
    evaluate_models(
        ['./models/littlePrince-iter25.m', './models/littlePrince-iter4.m', './models/littlePrince-average.m'],
        './data/Little-Prince-amr-bank/test/test.txt')
    evaluate_models_by_basic('./models/littlePrince','./data/Little-Prince-amr-bank/dev/dev.txt')
    print "end"
    '''

    averaged_model = average_k_best(basic_model_name,dev_amr_file,k)
    (precision, recall, f_score) = evaluate_model_by_file(averaged_model,gold_amr_file)
    print "Precision: %.3f" % precision
    print "Recall: %.3f" % recall
    print "Document F-score: %.3f" % f_score


def average_k_best(basic_model_name,dev_amr_file,  k):
    constants.FLAG_COREF = False
    constants.FLAG_PROP = False
    constants.FLAG_DEPPARSER = 'stdconv+charniak'

    iter = 1
    current_model_file = basic_model_name + '-iter' + str(iter) + '.m'
    model_names = []
    while os.path.exists(current_model_file):
        model_names.append(current_model_file)
        iter += 1
        current_model_file = basic_model_name + '-iter' + str(iter) + '.m'

    constants.FLAG_COREF = False
    constants.FLAG_PROP = False
    constants.FLAG_DEPPARSER = 'stdconv+charniak'

    dev_instances = preprocess(dev_amr_file, START_SNLP=False, INPUT_AMR=True)

    scored_model_names =[]

    for current_model_file in model_names:
        print "Loading model: ", current_model_file
        model = Model.load_model(current_model_file)
        parser = Parser(model=model, oracle_type=DET_T2G_ORACLE_ABT, action_type='basic', verbose=0, elog=sys.stdout)
        print "parsing file for model: ", current_model_file
        span_graph_pairs, results = parser.parse_corpus_test(dev_instances, False)
        print "saving parsed file for model: ", current_model_file
        del parser, model
        gc.collect()
        gc.collect()
        gc.collect()
        write_parsed_amr(results, dev_instances, dev_amr_file, suffix='%s.parsed' % ('eval'))
        _,_,f1 = eval(dev_amr_file + '.eval.parsed', dev_amr_file)
        scored_model_names.append(F1Scored(current_model_file,f1))

    k_best_model_names = []

    for scored_model_name in get_k_best(scored_model_names,k):
        k_best_model_names.append(scored_model_name.model_name)
    return average(k_best_model_names, basic_model_name+'-averaged.m')

def get_k_best(scored_model_names, k):
    q = Q.PriorityQueue()
    for scored_model in scored_model_names:
        q.put(scored_model)
        if q.qsize() > k:
            q.get()
    result = []
    while not q.empty():
        result.append(q.get())
    return result


def test_queue():
    scored_model_names = []
    scored_model_names.append(F1Scored('./models/littlePrince-iter1.m', 0.565))
    scored_model_names.append(F1Scored('./models/littlePrince-iter2.m', 0.572))
    scored_model_names.append(F1Scored('./models/littlePrince-iter3.m', 0.575))
    scored_model_names.append(F1Scored('./models/littlePrince-iter4.m', 0.580))
    scored_model_names.append(F1Scored('./models/littlePrince-iter5.m', 0.579))
    scored_model_names.append(F1Scored('./models/littlePrince-iter6.m', 0.576))
    scored_model_names.append(F1Scored('./models/littlePrince-iter7.m', 0.573))
    scored_model_names.append(F1Scored('/models/littlePrince-iter8.m', 0.572))
    scored_model_names.append(F1Scored('./models/littlePrince-iter9.m', 0.571))
    scored_model_names.append(F1Scored('./models/littlePrince-iter10.m', 0.570))
    scored_model_names.append(F1Scored('./models/littlePrince-iter11.m', 0.567))
    scored_model_names.append(F1Scored('./models/littlePrince-iter12.m', 0.569))
    scored_model_names.append(F1Scored('./models/littlePrince-iter13.m', 0.568))
    scored_model_names.append(F1Scored('./models/littlePrince-iter14.m', 0.569))
    scored_model_names.append(F1Scored('./models/littlePrince-iter15.m', 0.567))
    scored_model_names.append(F1Scored('./models/littlePrince-iter16.m', 0.564))
    scored_model_names.append(F1Scored('./models/littlePrince-iter17.m', 0.566))
    scored_model_names.append(F1Scored('./models/littlePrince-iter18.m', 0.566))
    scored_model_names.append(F1Scored('./models/littlePrince-iter19.m', 0.564))
    scored_model_names.append(F1Scored('./models/littlePrince-iter20.m', 0.563))
    scored_model_names.append(F1Scored('./models/littlePrince-iter21.m', 0.563))
    scored_model_names.append(F1Scored('./models/littlePrince-iter22.m', 0.563))
    scored_model_names.append(F1Scored('./models/littlePrince-iter23.m', 0.563))
    scored_model_names.append(F1Scored('./models/littlePrince-iter24.m', 0.563))
    scored_model_names.append(F1Scored('./models/littlePrince-iter25.m', 0.562))

    scored_model_names = get_k_best(scored_model_names, 5)
    for scored_model_name in scored_model_names:
        print 'el: ', scored_model_name


class F1Scored(object):
    def __init__(self, model_name, score):
        self.model_name = model_name
        self.score = score

    def __cmp__(self, other):
        return cmp(self.score, other.score)

    def __str__(self):
        return self.model_name + ', ' + str(self.score)


def evaluate_models_by_basic(model_basic_name, gold_amr_file):
    constants.FLAG_COREF = False
    constants.FLAG_PROP = False
    constants.FLAG_DEPPARSER = 'stdconv+charniak'

    iter = 1
    current_model_file = model_basic_name + '-iter' + str(iter) + '.m'
    model_names = []
    while os.path.exists(current_model_file):
        model_names.append(current_model_file)
        iter += 1
        current_model_file = model_basic_name + '-iter' + str(iter) + '.m'
    evaluate_models(model_names, gold_amr_file)


def evaluate_models(model_names, gold_amr_file):
    constants.FLAG_COREF = False
    constants.FLAG_PROP = False
    constants.FLAG_DEPPARSER = 'stdconv+charniak'

    test_instances = preprocess(gold_amr_file, START_SNLP=False, INPUT_AMR=True)

    for current_model_file in model_names:
        print "Loading model: ", current_model_file
        model = Model.load_model(current_model_file)
        parser = Parser(model=model, oracle_type=DET_T2G_ORACLE_ABT, action_type='basic', verbose=0, elog=sys.stdout)
        print "parsing file for model: ", current_model_file
        span_graph_pairs, results = parser.parse_corpus_test(test_instances, False)
        print "saving parsed file for model: ", current_model_file
        del parser, model
        gc.collect()
        gc.collect()
        gc.collect()
        write_parsed_amr(results, test_instances, gold_amr_file, suffix='%s.parsed' % ('eval'))
        evaluate(gold_amr_file + '.eval.parsed', gold_amr_file)
    print "Done!"


def evaluate_model_by_file(model, gold_amr_file):
    constants.FLAG_COREF = False
    constants.FLAG_PROP = False
    constants.FLAG_DEPPARSER = 'stdconv+charniak'

    test_instances = preprocess(gold_amr_file, START_SNLP=False, INPUT_AMR=True)
    parser = Parser(model=model, oracle_type=DET_T2G_ORACLE_ABT, action_type='basic', verbose=0, elog=sys.stdout)
    span_graph_pairs, results = parser.parse_corpus_test(test_instances, False)
    write_parsed_amr(results, test_instances, gold_amr_file, suffix='%s.parsed' % ('eval'))
    return eval(gold_amr_file + '.eval.parsed', gold_amr_file)


def evaluate(parsed_file, gold_file):
    (precision, recall, f_score) = eval(parsed_file, gold_file)
    print "Precision: %.3f" % precision
    print "Recall: %.3f" % recall
    print "Document F-score: %.3f" % f_score


def average(model_names, average_model_name):
    print "Init average model with ", model_names[0]
    average_model = Model.load_model(model_names[0])
    n = len(model_names)
    for i in range(1, n):
        gc.collect()
        print "Summing with ", model_names[i]
        new_model = Model.load_model(model_names[i])
        for j in range(0, 9):
            average_model.aux_weight[j] += new_model.aux_weight[j]
        for j in range(0, 9):
            average_model.avg_weight[j] += new_model.avg_weight[j]
        for j in range(0, 9):
            average_model.weight[j] += new_model.weight[j]

        del new_model
        gc.collect()
    print "Averaging..."
    for j in range(0, 9):
        average_model.aux_weight[j] /= n
    for j in range(0, 9):
        average_model.avg_weight[j] /= n
    for j in range(0, 9):
        average_model.weight[j] /= n
    print "Saving..."
    average_model.elog = sys.stdout
    average_model.save_model(average_model_name)
    return average_model

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='hello')
    arg_parser.add_argument('-b','--basic',default='./models/littlePrince',type=str,help='basic model name')
    arg_parser.add_argument('-d','--dev',default='./data/Little-Prince-amr-bank/dev/dev.txt',type=str,help='dev amr file')
    arg_parser.add_argument('-r','--test',default='./data/Little-Prince-amr-bank/test/test.txt',type=str,help='test amr file')
    arg_parser.add_argument('-k','--k',default= 4,type=int,help='models to average')

    args = arg_parser.parse_args()

    main(args)