import argparse
import pusion as p
import numpy as np
from sklearn.neural_network import MLPClassifier
import csv
import sys
sys.path.insert(0, '/home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/')
from model.utils import *


eval_metrics = [
    p.PerformanceMetric.ACCURACY,
    p.PerformanceMetric.MICRO_F1_SCORE,
    p.PerformanceMetric.MICRO_PRECISION,

    # p.PerformanceMetric.MICRO_RECALL,
    # p.PerformanceMetric.MICRO_F2_SCORE,
    # p.PerformanceMetric.MICRO_JACCARD_SCORE,
    # p.PerformanceMetric.MACRO_PRECISION,
    # p.PerformanceMetric.MACRO_RECALL,
    # p.PerformanceMetric.MACRO_F1_SCORE,
    # p.PerformanceMetric.MACRO_F2_SCORE,
    # p.PerformanceMetric.MACRO_JACCARD_SCORE,
    # p.PerformanceMetric.MEAN_MULTILABEL_ACCURACY,
    # p.PerformanceMetric.MEAN_CONFIDENCE,
    # p.PerformanceMetric.BALANCED_MULTICLASS_ACCURACY_SCORE
]


def fusion(dataset, y_true_path, y_lgbp_dir_path, y_slowfast_dir_path, output_path, slowfast_duplications):
    # get csv files paths
    if y_lgbp_dir_path is not None:
        # svm saves the csv files in one folder (for example: /SAMM/y_prob/C_1.0__kernel_rbf.csv)
        y_lgbp_paths_list = list(paths.list_files(y_lgbp_dir_path))
    # slowfast saves the csv files in one folder (for example: /<date>/SLOWFAST-ALPHA_4/y_prob_output)
    y_slowfast_paths_list = list(paths.list_files(y_slowfast_dir_path))
    y_slowfast_paths_list = [path for path in y_slowfast_paths_list if "y_prob_output.csv" in path]

    # create dictionary of paths
    fusion_paths = dict()
    fusion_paths['y_true_path'] = [y_true_path]
    if y_lgbp_dir_path is not None:
        fusion_paths['y_lgbp_path'] = y_lgbp_paths_list

    # create empty list for permutations
    permutations_dicts = []

    for i in range(slowfast_duplications):
        fusion_paths[f'y_slowfast_{i}_path'] = y_slowfast_paths_list

        # add permutations for the path so far, so if the requestd slowfasts is n, we'll have also n-1, n-2 etc
        permutations_dicts += create_permutations_from_dict(fusion_paths)


    # for each permutation, load the csv, get predictions and perform fusion
    for perm in tqdm(permutations_dicts):
        # default None for lgbp
        y_lgbp = None

        # load predictions
        if y_lgbp_dir_path is not None:
            y_test, y_lgbp, y_slowfasts_dict = load_lgbp_slowfast_and_labels(**perm)
        else:
            y_test, y_slowfasts_dict = load_slowfast_and_labels(**perm)

        # fuse predictions
        generic_matrix, ensemble_matrix = perform_fusion(y_test, y_lgbp, y_slowfasts_dict)

        # save report
        is_lgbp_exists = y_lgbp is not None
        save_report(output_path, generic_matrix, ensemble_matrix, slowfast_duplications, is_lgbp_exists=is_lgbp_exists, **perm)


def load_lgbp_slowfast_and_labels(y_true_path, y_lgbp_path, **y_slowfast_paths):
    # load predictions
    y_lgbp = load_lgbp(y_lgbp_path)
    y_slowfasts_dict = load_slowfast(**y_slowfast_paths)
    y_test = load_labels(y_true_path, y_lgbp)

    return y_test, y_lgbp, y_slowfasts_dict

def load_slowfast_and_labels(y_true_path, **y_slowfast_paths):
    # load predictions
    y_slowfasts_dict = load_slowfast(**y_slowfast_paths)
    random_y_of_slowfast = list(y_slowfasts_dict.values())[0]
    y_test = load_labels(y_true_path, random_y_of_slowfast)

    return y_test, y_slowfasts_dict


def load_lgbp(y_lgbp_path):
    y_lgbp = load_results_and_predict(y_lgbp_path, header='infer', index_col=0)
    return y_lgbp

def load_slowfast(**y_slowfast_paths):
    y_slowfasts_dict = {}

    for slowfast_i, y_path in y_slowfast_paths.items():
        y_slowfast_i = load_results_and_predict(y_path, header=None, index_col=None)
        y_slowfasts_dict[slowfast_i] = y_slowfast_i

    return y_slowfasts_dict

def load_labels(y_true_path, y):
    y_test = load_dataset_csv_and_get_labels(y_true_path, y.shape)

    return y_test

def perform_fusion(y_test, y_lgbp, y_slowfasts_dict):
    # create instances with names
    # y_lgbp =  "LGBP Predictions using SVM"
    # y_slowfast.name = "SlowFast Predictions"

    # create instances. first is LGBP, the rest are SlowFast according to dict length
    if y_lgbp is not None:
        instances = ["LGBP"] + [f"SlowFast_{i}" for i in range(len(y_slowfasts_dict.keys()))]
    else:
        instances = [f"SlowFast_{i}" for i in range(len(y_slowfasts_dict.keys()))]

    # Create a numpy tensor
    y_ensemble = []
    if y_lgbp is not None:
        y_ensemble += [y_lgbp]
    y_ensemble += [y_slowfast for slowfast_i, y_slowfast in y_slowfasts_dict.items()]
    y_ensemble = np.array(y_ensemble)

    # perform generic fusion to get some statistics
    generic_matrix, ensemble_matrix = generic_fusion(y_test, y_ensemble, instances)

    # perform borda count off fusion
    # borda_matrix = borda_fusion(y_test, y_ensemble)

    return generic_matrix, ensemble_matrix


def save_report(output_path, generic_matrix, ensemble_matrix, total_number_of_slowfasts, is_lgbp_exists=False, **paths):
    # get names
    if is_lgbp_exists:
        lgbp_run = os.path.splitext(os.path.basename(paths['y_lgbp_path']))[0]
    slowfast_runs = [os.path.basename(os.path.dirname(os.path.normpath(paths[f'y_slowfast_{i}_path']))) if f'y_slowfast_{i}_path' in paths else None for i in range(total_number_of_slowfasts)]

    # create names columns
    slowfasts_names_dict = {f'SlowFast_{i}_name': slowfast_runs[i] for i in range(total_number_of_slowfasts)}
    if is_lgbp_exists:
        names_dict = {'LGBP_name': lgbp_run, **slowfasts_names_dict}
    else:
        names_dict = {**slowfasts_names_dict}

    # get LGBP and SlowFast accuracy
    ensemble_records = {x.replace(' ', ''): v for x, v in ensemble_matrix.records.items()}
    if is_lgbp_exists:
        LGBP_accuracy = ensemble_records['LGBP'][0]
    SlowFast_accuracies = [ensemble_records[f'SlowFast_{i}'][0] if f'SlowFast_{i}' in ensemble_records else None for i in range(total_number_of_slowfasts)]

    #build accuracies list
    accuracies = []
    if is_lgbp_exists:
        accuracies += [LGBP_accuracy]
    accuracies += SlowFast_accuracies
    accuracies = list(filter(None, accuracies))

    # create accuracies columns
    slowfasts_acc_dict = {f'SlowFast_{i}_accuracy': SlowFast_accuracies[i] for i in range(total_number_of_slowfasts)}
    if is_lgbp_exists:
        acc_dict = {'LGBP_accuracy': LGBP_accuracy, **slowfasts_acc_dict}
    else:
        acc_dict = {**slowfasts_acc_dict}


    # merge
    row_data = {**names_dict, **acc_dict}

    # add Number_Of_Winning_Fusions
    row_data['Number_Of_Winning_Fusions'] = 0

    # there are too much trailing white spaces at these strings
    generic_records = {x.replace(' ', ''): v for x, v in generic_matrix.records.items()}

    for i in range(len(generic_matrix.instance_names)):
        name = generic_matrix.instance_names[i].strip()
        accuracy = generic_records[name][0]
        row_data[f'{name}_accuracy'] = accuracy

        # check if the current fusion score is higher than the models scores
        if all(accuracy > x for x in accuracies):
            row_data['Number_Of_Winning_Fusions'] += 1

    # create accuracy report csv file
    create_accuracy_file_at_runtime(output_path, row_data, print_log=False)


def generic_fusion(y_test, y_ensemble, instances):
    # Initialize the general framework interface
    conf = p.Configuration(
        method=p.Method.GENERIC
    )

    # Initialize the general framework interface
    dp = p.DecisionProcessor(conf)

    # Fuse the ensemble classification outputs (test dataset)
    dp.combine(y_ensemble)

    # print metrics
    generic_matrix, ensemble_matrix = print_generic_fusion_metrics(dp, y_test, y_ensemble, instances)

    return generic_matrix, ensemble_matrix


def borda_fusion(y_test, y_ensemble):
    # User defined configuration
    conf = p.Configuration(
        method=p.Method.BORDA_COUNT,
        problem=p.Problem.MULTI_CLASS,
        assignment_type=p.AssignmentType.CONTINUOUS,
        coverage_type=p.CoverageType.COMPLEMENTARY_REDUNDANT
    )

    # Initialize the general framework interface
    dp = p.DecisionProcessor(conf)

    # Fuse the ensemble classification outputs (test dataset)
    y_comb = dp.combine(y_ensemble)

    # print metrics
    borda_matrix = print_single_fusion_metrics(dp, y_test, y_comb)

    return borda_matrix


def print_generic_fusion_metrics(dp, y_test, y_ensemble, instances):
    # print("============= Ensemble ===============")
    # print("ndarray[0] is LGBP, ndarray[1] is Slowfast")
    eval_classifiers = p.Evaluation(*eval_metrics)
    eval_classifiers.set_instances(instances)
    eval_classifiers.evaluate(y_test, y_ensemble)
    # print(eval_classifiers.get_report())
    ensemble_matrix = eval_classifiers.get_report()

    # print()
    eval_combiner = p.Evaluation(*eval_metrics)
    eval_combiner.set_instances(dp.get_combiners())
    eval_combiner.evaluate(y_test, dp.get_multi_combiner_decision_output())
    dp.set_evaluation(eval_combiner)
    # print(dp.report())
    generic_matrix = dp.evaluation.get_report()

    return generic_matrix, ensemble_matrix


def print_single_fusion_metrics(dp, y_test, y_comb):
    print("============== Borda Count Combiner ==============")
    eval_combiner = p.Evaluation(*eval_metrics)
    eval_combiner.set_instances(dp.get_combiner())
    eval_combiner.evaluate(y_test, y_comb)
    print(eval_combiner.get_report())
    borda_matrix = eval_combiner.performance_matrix

    return borda_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path arguments
    parser.add_argument('--dataset', type=str,  # required=True,
                        default="SMIC",
                        help='Name of dataset.')
    parser.add_argument('--y_true_path', type=str, required=True,
                        # default="/home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/classes/2022-08-17_08:00_SMIC_Split/SMIC/test.csv",
                        help='Path to load true labels.')
    parser.add_argument('--y_lgbp_dir_path', type=str, # required=True,
                        default=None,
                        # default="/home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/SVM/2022-06-23_13:00/SAMM/y_prob/",
                        help='Path to a directory with lgbp predictions csv files.')
    parser.add_argument('--y_slowfast_dir_path', type=str, required=True,
                        # default="/home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/SlowFast/2022-08-23_10:00_SMIC/",
                        help='Path to a directory with SlowFast predictions directories.')
    parser.add_argument('--output_path', type=str,  required=True,
                        # default="/home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/Fusion/2022-08-23_10:00_SMIC/",
                        help='Path to save the fusion results')
    parser.add_argument('--slowfast_duplications', type=int,  # required=True,
                        default=1,
                        help='fuse LGBP with number of slowfasts')
    # parser.add_argument('--weights', type=str,  # required=True,
    #                     default="80/10/10",
    #                     help='How to divide wieghts between models first weight is LGBP, the rest are slowfast. 80/10/10 or 80/10/5/5')

    # parse arguments
    args = parser.parse_args()


    # create parameters for data making
    fusion_params = {
        'dataset': args.dataset,
        'y_true_path': args.y_true_path,
        'y_lgbp_dir_path': args.y_lgbp_dir_path,
        'y_slowfast_dir_path': args.y_slowfast_dir_path,
        'output_path': args.output_path,
        'slowfast_duplications': args.slowfast_duplications,
        # 'weights_list': args.weights,
    }

    # run functions
    fusion(**fusion_params)
