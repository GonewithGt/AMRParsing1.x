import os

import gc

import constants
from constants import DET_T2G_ORACLE_ABT
from parser import Parser
from smatch_v1_0.smatch_modified import *
from model import Model
from amr_parsing import preprocess,write_parsed_amr
reload(sys)
sys.setdefaultencoding('utf-8')

def main():
    '''
    average(['./models/littlePrince-iter1.m', './models/littlePrince-iter25.m'], './models/littlePrince-avg.m')
    evaluate('./data/Little-Prince-amr-bank/test/test.txt.all.parsed', './data/Little-Prince-amr-bank/test/test.txt')
    '''
    evaluate_models('./models/littlePrince','./data/Little-Prince-amr-bank/test/test.txt')
    print "end"


def evaluate_models(model_basic_name, gold_amr_file):
    constants.FLAG_COREF= False
    constants.FLAG_PROP= False
    constants.FLAG_DEPPARSER= 'stdconv+charniak'

    iter = 1
    current_model_file = model_basic_name + '-iter' + str(iter) + '.m'
    model_names = []
    while os.path.exists(current_model_file):
        model_names.append(current_model_file)
        iter += 1
        current_model_file = model_basic_name + '-iter' + str(iter) + '.m'

    test_instances = preprocess(gold_amr_file, START_SNLP=False, INPUT_AMR=True)

    for current_model_file in model_names:
        print "Loading model: ", current_model_file
        model = Model.load_model(current_model_file)
        parser = Parser(model=model,oracle_type=DET_T2G_ORACLE_ABT,action_type= 'basic',verbose= 0,elog=sys.stdout)
        print "parsing file for model: ", current_model_file
        span_graph_pairs,results = parser.parse_corpus_test(test_instances,False)
        print "saving parsed file for model: ", current_model_file
        del parser, model
        gc.collect()
        gc.collect()
        gc.collect()
        write_parsed_amr(results,test_instances,gold_amr_file,suffix='%s.parsed'%('eval'))
        evaluate(gold_amr_file+'.eval.parsed',gold_amr_file)
    print "Done!"




def evaluate( parsed_file,gold_file):
    (precision, recall, best_f_score) = eval(parsed_file,gold_file)
    print "Precision: %.3f" % precision
    print "Recall: %.3f" % recall
    print "Document F-score: %.3f" % best_f_score


def average(model_names, average_model_name):
    average_model = Model.load_model(model_names[0])
    n = len(model_names)
    for i in range(1, n):
        new_model = Model.load_model(model_names[i])
        for j in range(0, 9):
            average_model.aux_weight[j] = average_model.aux_weight[j] + new_model.aux_weight[j]
        for j in range(0, 9):
            average_model.avg_weight[j] = average_model.avg_weight[j] + new_model.avg_weight[j]
        for j in range(0, 9):
            average_model.weight[j] = average_model.weight[j] + new_model.weight[j]

    for j in range(0, 9):
        average_model.aux_weight[j] /= n
    for j in range(0, 9):
        average_model.avg_weight[j] /= n
    for j in range(0, 9):
        average_model.weight[j] /= n
    average_model.save_model(average_model, average_model_name)


if __name__ == "__main__":
    main()
