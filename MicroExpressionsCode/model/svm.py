import matplotlib.pyplot as plt
import pandas as pd
import itertools
import os
import sys
import csv
import time
import numpy as np
from tqdm import tqdm
np.set_printoptions(linewidth=400)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

sys.path.insert(0, '/home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/')
from model.utils import *

# region Global Variables

TEST_SIZE = 0.5

#endregion


# Getting the data

def svm(X_train, X_test, y_train, y_test, svm_params, output_path, class_mapping):
    # create list of of dictionary with all permutations
    permutations_dicts = create_permutations_from_dict(svm_params)
    existing_tests = get_accuracy_file_as_dict(output_path, 'accuracy_score')
    results = []

    print(f"[INFO] Total number of parameters permutations is {len(permutations_dicts)}")
    for params in tqdm(permutations_dicts):
        try:
            # stringed_params = params.copy()
            # stringed_params['C'] = str(stringed_params['C'])
            # if stringed_params['gamma'] in ['1.0']:
            #     stringed_params['gamma'] = str(int(stringed_params['gamma']))
            # stringed_params['degree'] = str(stringed_params['degree'])
            # if stringed_params in existing_tests:
            #     print(f"[INFO] parameters already tested. skipping the following permutation: {params}")
            #     continue
            try:
                temp = float(params['gamma'])
                params['gamma'] = temp
            except:
                pass
            params['gamma'] = params['gamma']
            print(f"running job with paramters: {params}")
            # Training the model
            clf = SVC(**params, probability=True)
            clf.fit(X_train, y_train)

            # Making predictions
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)

            save_metrics(params, X_test, y_test, y_pred, y_prob, output_path, class_mapping)

            results.append((params, y_pred, y_prob))
        except Exception as e:
            print("Iteration failed with the following exception:")
            print(e)
            continue
        



    # parameters = {
    #     'C': [10, 100, 1000],  # [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    #     'kernel': ['linear', 'sigmoid'],  # ['linear', 'poly', 'rbf', 'sigmoid']
    #     'gamma': ['scale', 'auto', 0.1, 1, 10, 100],
    #     # 'degree': [0, 1, 2, 3, 4, 5, 6],
    # }
    # svc = SVC()
    # clf_gs = GridSearchCV(svc, parameters, verbose=3)
    # clf_gs.fit(X_train, y_train)
    #
    # print("Best parameters set found on development set:")
    # print()
    # print(clf_gs.best_params_)
    # print()
    # print("Grid scores on development set:")
    # print()
    # means = clf_gs.cv_results_["mean_test_score"]
    # stds = clf_gs.cv_results_["std_test_score"]
    # for mean, std, params in zip(means, stds, clf_gs.cv_results_["params"]):
    #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    # print()

    return results


def save_metrics(params, X_test, y_test, y_pred, y_prob, output_path, class_mapping):
    params_string = str(params).replace(" ", "_").replace(":", "").replace("{", "").replace("}", "").replace("'", "").replace(",", "_")

    # show the classification report
    report = classification_report(y_test, y_pred, labels=range(0, len(class_mapping.keys())), target_names=class_mapping.keys(), zero_division=1)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print("\n" + report)
    print(cm)
    print("\n" + "Accuracy Score: " + str(acc))

    if output_path is not None:
        y_prob_path = os.path.join(output_path, "y_prob")
        y_prob_file_path = os.path.join(output_path, "y_prob", params_string + ".csv")
        print(f"[INFO] saving y_prob to {y_prob_file_path}")
        res = pd.DataFrame(y_prob)

        # create y_prob file
        if not os.path.exists(y_prob_path):
            os.makedirs(y_prob_path, exist_ok=True)
        res.to_csv(y_prob_file_path)

        row_data = params
        row_data['accuracy_score'] = acc

        # create accuracy report csv file
        create_accuracy_file_at_runtime(output_path, row_data)

