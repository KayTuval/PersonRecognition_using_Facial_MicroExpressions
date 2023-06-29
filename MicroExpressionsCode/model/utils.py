import csv
import os
from tqdm import tqdm
import cv2
import numpy as np
from imutils import paths
import json
import pandas as pd
import itertools
from dask import dataframe as dd

# region Functions

def detect_face(img):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    # Detect face
    face = face_cascade.detectMultiScale(img, minNeighbors=4)

    return face


def crop_face(img, face):
    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in face:
        img_cropped = img[y:y + h, x:x + w]

    return img_cropped


def load_face_dataset(input_path, lbp=False, min_samples=1, dataset='', flatten=False, class_mapping=dict()):
    # grab the paths to all images in our input directory, extract
    # the name of the person (i.e., class label) from the directory
    if lbp:
        data_paths = list(paths.list_files(input_path))
    else:
        data_paths = list(paths.list_images(input_path))

    # filter out only data in dataset
    data_paths = [path for path in data_paths if dataset in path]

    # filter out only data in class mapping
    # os.path.basename(os.path.normpath(os.path.join(path, os.pardir))) --> last folder name (for example SAMM_20)
    data_paths = [path for path in data_paths if os.path.basename(os.path.normpath(os.path.join(path, os.pardir))) in class_mapping.keys()]

    names = [p.split(os.path.sep)[-2] for p in data_paths]
    (names, counts) = np.unique(names, return_counts=True)
    names = names.tolist()

    # initialize lists to store our extracted faces and associated
    # labels
    faces = []
    labels = []
    paths_lst = []

    # loop over the image paths
    for data_path in tqdm(data_paths):
        # load the image from disk and extract the name of the person
        # from the subdirectory structure
        if lbp:
            data = pd.read_csv(data_path, header=None)
            data = np.ravel(data)
            # data = data[:100]  # DEBUG
            # data = np.asarray([0])  # DEBUG
        else:
            data = cv2.imread(data_path, 0)
            if flatten:
                data = data.flatten()

        name = data_path.split(os.path.sep)[-2]

        # only process images that have a sufficient number of
        # examples belonging to the class
        if counts[names.index(name)] < min_samples:
            continue

        # update our faces and labels lists
        faces.append(data)
        labels.append(name)
        paths_lst.append(data_path)

    # convert our faces and labels lists to NumPy arrays
    faces = np.array(faces)
    labels = np.array(labels)
    paths_lst = np.array([i.split("/")[-1].split(".")[0] for i in list(paths_lst)])

    # return a 2-tuple of the faces and labels
    return (faces, labels, paths_lst)


def load_classes_ids(path):
    with open(path) as json_file:
        class_mapping = json.load(json_file)
    return class_mapping


def encode_faces_with_classesids_json(labels, class_mapping):
    labels_ids = [class_mapping[i] for i in tqdm(labels)]
    labels_ids = np.array(labels_ids)
    return labels_ids


def load_csv_and_return_X_y(path):
    df = pd.read_csv(path, header=None)
    X = [i.split("/")[-1].split(".")[0] for i in list(df[0])]
    y = list(df[1])
    return X, y


def load_train_test_split(faces, paths_lst, csv_path):
    # get paths to csv files
    train_path = os.path.join(csv_path, "train.csv")
    val_path = os.path.join(csv_path, "val.csv")
    test_path = os.path.join(csv_path, "test.csv")

    # load csv files
    X_train, y_train = load_csv_and_return_X_y(train_path)
    X_val, y_val = load_csv_and_return_X_y(val_path)
    X_test, y_test = load_csv_and_return_X_y(test_path)

    X_train, y_train = X_train + X_val,  y_train + y_val

    # replace file name with the face itself
    paths_and_faces_dictionary = dict(zip(paths_lst, faces))
    X_train = [paths_and_faces_dictionary[i] for i in X_train]
    X_test = [paths_and_faces_dictionary[i] for i in X_test]

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test


def load_results_and_predict(path, header, index_col):
    df = pd.read_csv(path, header=header, index_col=index_col)
    preds = np.zeros_like(df.values)
    preds[np.arange(len(df)), df.values.argmax(1)] = 1
    return preds


def load_dataset_csv_and_get_labels(path, shape):
    df = pd.read_csv(path, header=None)
    labels = np.zeros(shape)
    labels[np.arange(len(df[1])), df[1]] = 1
    return labels


def create_permutations_from_dict(dictionary):
    keys, values = zip(*dictionary.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_dicts


def get_accuracy_file_as_dict(output_path, row_to_exclude, print_logs=True):
    if not os.path.exists(output_path):
        return []
    acc_path = os.path.join(output_path, "accuracy_scores.csv")
    if not os.path.exists(acc_path):
        return []
    if print_logs:
        print(f"[INFO] There's already an `accuracy_scores.csv` file in tht directory. going to load it...")
    with open(acc_path, "r") as f:
        csv_reader = csv.DictReader(f)
        name_records = list(csv_reader)
        [row.pop(row_to_exclude) for row in name_records]
        return name_records


def create_accuracy_file_at_runtime(output_path, row_data, print_log=True):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # create accuracy file
    acc_path = os.path.join(output_path, "accuracy_scores.csv")
    if print_log:
        print(f"[INFO] saving accuracy to {acc_path}")
    write_header = False
    if not os.path.exists(acc_path):
        write_header = True

    with open(acc_path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row_data)