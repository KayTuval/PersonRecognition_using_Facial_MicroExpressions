import time
import argparse
import os
import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import sys
sys.path.insert(0, '/home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/')

from main_dir.consts import *
from model.utils import *
from model.svm import *





def classifier(frames_path,
               classes_ids_json_path,
               output_path=None,
               dataset='',
               min_samples=1,
               lbp=True,
               split_data='load',
               csv_path='',
               svm_params=dict()
               ):
    # main script

    # Load json file with classes as dictionary
    print(f"[INFO] Loading classesids from {classes_ids_json_path}...")
    class_mapping = load_classes_ids(classes_ids_json_path)
    print(f"[INFO] Number of classes is {len(class_mapping)}")

    # load the faces
    print(f"[INFO] loading dataset from {frames_path}")
    (faces, labels, paths_lst) = load_face_dataset(frames_path, lbp=lbp, dataset=dataset, flatten=True, min_samples=min_samples, class_mapping=class_mapping)
    print("[INFO] {} data in dataset".format(len(faces)))

    # encode the string labels as integers
    print(f"[INFO] encoding data using classesids from {classes_ids_json_path}...")
    labels = encode_faces_with_classesids_json(labels, class_mapping)

    # construct our training and testing split
    if split_data == 'split':
        print("[INFO] splitting the dataset to train and test...")
        X_train, X_test, y_train, y_test = train_test_split(faces, labels, train_size=TEST_SIZE, random_state=42)#, stratify=labels)
    elif split_data == 'load':
        print(f"[INFO] loading the csv files to split train and test. all have same directory as {csv_path}")
        X_train, X_test, y_train, y_test = load_train_test_split(faces, paths_lst, csv_path)

    # train
    print(f"[INFO] training SVM face recognizer... Output path is {output_path}")
    start = time.time()
    results = svm(X_train, X_test, y_train, y_test, svm_params, output_path, class_mapping)
    end = time.time()
    print("[INFO] training and predicting took {:.4f} seconds".format(end - start))

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path arguments
    parser.add_argument('--frames_path', default=SLOWFAST_LBP_DATA_PATH, type=str,
                        help='Path to load data. Can also be LBP for shortcut')
    # parser.add_argument('--classes_ids_json_path', default='/home/khen_proj_1/PycharmProjects/slowfast_test/slowfast/data/docs/SAMM/classids.json', type=str,
    parser.add_argument('--classes_ids_json_path', default=CLASSIDS_JSON_PATH, type=str,
                        help='Path to load classes ids')
    parser.add_argument('--csv_path', default=SLOWFAST_DATA_PATH, type=str,
                        help='Path to slowfast train csv file')
    parser.add_argument('--output_path', default=SVM_OUTPUT_PATH, type=str,
                        help='Path to save probabilities of y: y_prob')
    parser.add_argument('--output_folder_name', default=str(str(datetime.datetime.now())), type=str,
                        help='Name of folder in output_path')

    # Preprocess arguments
    parser.add_argument('--dataset', type=str, default='SAMM',  # required=True,
                        help='SAMM, CASME2 or SMIC')
    parser.add_argument('--min_samples', default=1, type=int,
                        help='Use subject that have a minimum of samples')
    parser.add_argument('--lbp', default=True, type=bool,
                        help='Whether to load lbp')
    parser.add_argument('--split_data', default='load', type=str,
                        help='Whether to split the data or load it')

    # SVM arguments
    parser.add_argument('--C', default=1.0, type=float, nargs='*',
                        help='Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.')
    parser.add_argument('--kernel', default='rbf', type=str, nargs='*',
                        help='Specifies the kernel type to be used in the algorithm. {linear, poly, rbf, sigmoid, precomputed}')
    parser.add_argument('--gamma', default='scale', type=str, nargs='*',
                        help='Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. {scale, auto}')
    parser.add_argument('--degree', default='3', type=int, nargs='*',
                        help='Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.')

    # parse arguments
    args = parser.parse_args()

    # data path shortcut
    if args.frames_path == 'LBP':
        frames_path = SLOWFAST_LBP_DATA_PATH
    else:
        frames_path = args.frames_path


    # create svm parameters, all must be lists
    svm_params = {
        'C': args.C if isinstance(args.C, list) else [args.C],
        'kernel': args.kernel if isinstance(args.kernel, list) else [args.kernel],
        'gamma': args.gamma if isinstance(args.gamma, list) else [args.gamma],
        'degree': args.degree if isinstance(args.degree, list) else [args.degree],
    }

    # Join output path and output folder
    output_path_full = os.path.join(args.output_path, args.output_folder_name, args.dataset)

    # create parameters for data making
    classifier_params = {
        'frames_path': frames_path,
        'classes_ids_json_path': args.classes_ids_json_path,
        'csv_path': args.csv_path,
        'output_path': output_path_full,
        'dataset': args.dataset,
        'min_samples': args.min_samples,
        'lbp': args.lbp,
        'split_data': args.split_data,
        'svm_params': svm_params
    }

    # create permutations of paramteres
    permutations_dicts = create_permutations_from_dict(svm_params)
    print()
    print(f"[INFO] Going to load data and then perform SVM on {len(permutations_dicts)} permutations of the data")
    print()

    # run classifier
    classifier(**classifier_params)



    # # generate a sample of testing data
    # idxs = np.random.choice(range(0, len(testY)), size=10, replace=False)
    #
    # # inverse class dictionary
    # inverse_class_mappings = {value: key for key, value in class_mapping.items()}
    #
    # # loop over a sample of the testing data
    # for i in tqdm(idxs):
    #     # grab the predicted name and actual name
    #     predName = inverse_class_mappings[([predictions[i]])[0]]
    #     actualName = inverse_class_mappings[testY[i]]
    #
    #     # grab the face image and resize it such that we can easily see
    #     # it on our screen
    #     face = np.dstack([testX[i]] * 3)
    #     face = imutils.resize(face, width=250)
    #
    #     # draw the predicted name and actual name on the image
    #     cv2.putText(face, "pred: {}".format(predName), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    #     cv2.putText(face, "actual: {}".format(actualName), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    #
    #     # display the predicted name, actual name, and confidence of the
    #     # prediction (i.e., chi-squared distance; the *lower* the distance
    #     # is the *more confident* the prediction is)
    #     print("[INFO] prediction: {}, actual: {}, confidence: {:.2f}".format(predName, actualName, confidence[i]))
    #
    #     # display the current face to our screen
    #     cv2.imshow("Face", face)
    #     cv2.waitKey(0)
    #
    #





    # if model == "knn":
    #     # train our LBP face recognizer using KNN
    #     print("[INFO] training KNN face recognizer...")
    #     recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
    #     start = time.time()
    #     recognizer.train(X_train, y_train)
    #     end = time.time()
    #     print("[INFO] training took {:.4f} seconds".format(end - start))
    #
    #     # initialize the list of predictions and confidence scores
    #     print("[INFO] gathering predictions...")
    #     predictions = []
    #     confidence = []
    #     start = time.time()
    #
    #     # loop over the test data
    #     for i in tqdm(range(0, len(X_test))):
    #         # classify the face and update the list of predictions and
    #         # confidence scores
    #         (prediction, conf) = recognizer.predict(X_test[i])
    #         predictions.append(prediction)
    #         confidence.append(conf)
    #
    #     # measure how long making predictions took
    #     end = time.time()
    #     print("[INFO] inference took {:.4f} seconds".format(end - start))