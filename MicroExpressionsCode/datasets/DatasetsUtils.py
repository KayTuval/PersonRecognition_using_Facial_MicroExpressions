import pandas as pd
import numpy as np
import os
import re
import errno
import sys
import cv2
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from skimage import transform
import json

from datasets import DataAugmentation as aug

sys.path.insert(0, '/home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/')
from khen_hardware_libs.VideoClass import VideoClass
from main_dir.consts import *
from model.lbp import get_lgbp_histogram
from model.utils import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)





# region Functions

def get_unioned_df(args_dict):
    samm_df = get_df(**args_dict["SAMM"])
    casme2_df = get_df(**args_dict["CASME2"])
    smic_df = get_df(**args_dict["SMIC"])
    return pd.concat([samm_df, casme2_df, smic_df], keys=["SAMM", "CASME2", "SMIC"])


def get_df(path, remove_rows_under_minimum_frames_number=False, selected_subjects=[], split_multiple_au=False,
           remove_letters=False, reset_index=False, validate_files_exists=False, min_samples=1):
    df = pd.read_csv(path)
    # TODO: add function that handles the multiple ME in SMIC
    df = _remove_duplicate_columns(df, path)  # SMIC has several columns for different micro expressions
    # Consistency
    df = _remove_empty_columns(df, path)  # CASME2 has 2 empty columns, SMIC has 3
    df = _rename_columns_to_pascal(df, path)  # for consistency
    df = _drop_problematic_rows(df, path)  # currently only 1 row makes problems
    df = _add_apex_frame_column_for_smic(df,path)  # SMIC doesn't have ApexFrame column. let's take the middle between the Onset and Offset
    df = _convert_string_series_to_int(df)  # must come after _drop_problematic_rows()
    df = _add_columns_from_other_df(df, path)  # adding 'NumberOfFrames' to CASME2 and bunch of Nan for irrelevant columns
    df = _rename_subjects_name(df, path)  # to differentiate between subjects from each dataset

    # Emotions
    df = _get_accurate_emotions_in_smic(df, path)

    # filter unwanted data
    if remove_rows_under_minimum_frames_number:
        df = _remove_rows_under_minimum_frames_number(df)  # keep only rows with more than the minimum frames

    # remove all subject with less than `min_samples` samples
    v = df[['Subject']]
    samples_num = min_samples - 1
    df = df[v.replace(v.stack().value_counts()).gt(samples_num).all(samples_num)]

    # filter only selected subjects
    if selected_subjects:
        df = _filter_selected_subjects(df, selected_subjects)
    # Add Columns
    df = _calculate_frames_window_from_apex(df, path)  # for future consistency when giving the data to the model
    df = _add_paths_columns(df, path)  # for easy loading later on


    # Explode Rows
    df = _explode_action_units_combinations(df, split_multiple_au,
                                            remove_letters)  # some videos has multiple au activated, this function breaks the row and create a new row for every au
    df = _reset_index_if_needed(df,
                                reset_index)  # if entries were divided, the index remains the same, maybe we should re-index all

    if validate_files_exists:
        if _is_samm(path):
            dataset = "SAMM"
        elif _is_casme2(path):
            dataset = "CASME2"
        elif _is_smic(path):
            dataset = "SMIC"
        validate_frames_of_files_in_df(df, dataset)

    # ### debug
    # print("df.shape: ", df.shape)
    # # print(df.head(10))
    # print(df[['Subject', 'FileName', 'Emotion', 'SubjectAbsolutePath', 'SubjectRelativePath', 'MEAbsolutePath', 'MERelativePath']].head(10))
    # return
    # ### endDebug

    return df


def show_au_ranking(df):
    return df['ActionUnits'].value_counts()


def show_subject_ranking(df):
    return df['Subject'].value_counts()


def show_highest_used_au_in_df(df, AURank, topN):
    return df[df['ActionUnits'].isin(AURank.index[:topN])]


def validate_frames_of_files_in_df(df, dataset):
    print("looking for frames of dataset {dataset} needed by df".format(dataset=dataset))
    success = True
    if not _is_casme2(dataset) and not _is_samm(dataset) and not _is_smic(dataset):
        raise ValueError(UNKNOWN_PATH_ERROR)
    for index, row in df.iterrows():
        if not os.path.exists(row['SubjectAbsolutePath']):  # making sure the subject's folder exists
            success = False
            print(row)
            raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), row['SubjectAbsolutePath'])
        if not os.path.exists(row['MEAbsolutePath']):  # making sure the subjects video's folder exists
            success = False
            print(row)
            raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), row['MEAbsolutePath'])
        me_frames = sorted(os.listdir(row['MEAbsolutePath']))
        me_frames_numbers = [int(re.search(FRAMES_REGEX_PATTERN, frame).group(0)) for frame in me_frames]
        missing_frames = _find_missing(me_frames_numbers, row['OnsetFrame'], row['OffsetFrame'])
        if missing_frames:  # making sure all frames between Onset and Offset are preesent
            print("There are missing frames in {path}: {missing_frames}".format(path=row['MEAbsolutePath'],
                                                                                missing_frames=missing_frames))
            success = False
        for frame_number in me_frames_numbers:
            if not (int(row['OnsetFrame']) <= int(frame_number) <= int(row['OffsetFrame'])):
                pass
    if success:
        print("Success! All frames of dataset {dataset} needed by df are in place!".format(dataset=dataset))
        return True
    else:
        print("Failure! Some frames of dataset {dataset} needed by df are missing...".format(dataset=dataset))
        return False


def load_dataset(df, dataset, fps, validate_presence=True):
    validation_result = validate_frames_of_files_in_df(df, dataset)
    if not validation_result:
        return
    if _is_casme2(dataset):
        fps = CASME2_FPS
    elif _is_samm(dataset):
        fps = SAMM_FPS
    elif _is_smic(dataset):
        fps = SMIC_FPS

    index = 0
    for index, row in df.iterrows():
        print(index)
        ME_video = VideoClass.VideoClass()
        ME_video.load_video_from_folder_imgs(row['MEAbsolutePath'], fps)
        df.loc[index, 'VideoClass'] = ME_video
    print(f"finished loading {index + 1} videos to df")
    return df


def save_classes_ids_files(json_path, txt_path, class_ids, classes):
    directory_path = os.path.dirname(json_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    with open(json_path, 'w') as fp:
        json.dump(class_ids, fp, indent=2)

    directory_path = os.path.dirname(txt_path)
    if not os.path.exists(txt_path):
        os.makedirs(directory_path, exist_ok=True)
    with open(txt_path, 'w') as f:
        for idx, subject in enumerate(classes):
            if idx != len(classes) - 1:
                f.write("%s\n" % subject)
            else:
                f.write("%s" % subject)


def make_videos_of_microexpression_in_slowfast_directory(
        df,
        create_videos=True,
        augment=False,
        pad_frames=False,
        create_stills=False,
        create_lbp=False,
        create_classesids_files=False,
        copy_path=""
):
    # Checking if the directories for the data exists, if not, create them
    print("[INFO] Checking if data directories exists")
    check_if_directory_exist(SLOWFAST_DATA_PATH, create=True)
    if create_stills:
        check_if_directory_exist(SLOWFAST_FIRST_FRAMES_DATA_PATH, create=True)
    if create_lbp:
        check_if_directory_exist(SLOWFAST_LBP_DATA_PATH, create=True)

    # create classids.json and classes.txt
    print("[INFO] Creating cllassesids files")
    if create_classesids_files:
        subjects_ranking = show_subject_ranking(df)
        classes = subjects_ranking.index
        class_ids = dict()
        for idx, subject in enumerate(classes):
            class_ids[subject] = idx

        # save for SlowFast inputs
        save_classes_ids_files(CLASSIDS_JSON_PATH, CLASSES_TXT_PATH, class_ids, classes)

        # save for SVM and fusion inputs
        json_copy_path = os.path.join(copy_path, 'classids.json')
        txt_copy_path = os.path.join(copy_path, 'classes.txt')
        save_classes_ids_files(json_copy_path, txt_copy_path, class_ids, classes)

    df_new = df.copy()
    if create_videos or create_stills or create_lbp:
        for index, row in df.iterrows():  # iterate rows in df
            # Checking if the directories for the subjects data exists, if not, create them
            print(f"[INFO] Checking if subject {row['Subject']} directories exists")
            if create_videos:
                check_if_directory_exist(row['SlowFastSubjectRelativePath'], create=True)
            if create_stills:
                check_if_directory_exist(row['SlowFastSubjectFirstFramesRelativePath'], create=True)
            if create_lbp:
                check_if_directory_exist(row['SlowFastSubjectLbpRelativePath'], create=True)

            # set the image new dimensions
            if _is_samm(row['Subject']):
                dim = (SAMM_RESIZED_WIDTH, SAMM_RESIZED_HEIGHT)
            elif _is_casme2(row['Subject']):
                dim = (CASME2_RESIZED_WIDTH, CASME2_RESIZED_HEIGHT)
            elif _is_smic(row['Subject']):
                dim = (SMIC_RESIZED_WIDTH, SMIC_RESIZED_HEIGHT)

            # if we want to create videos
            if create_videos:
                # set path to save the video
                output_path = row['SlowFastSubjectRelativePath'] + row['FileName']
                video, _ = _create_VideoWriter_and_add_to_df(output_path, dim)

                # set path to save the augmented video and add rows to df for later splitting of test, train and val
                if augment:
                    video_angle_m30, df_new = _create_VideoWriter_and_add_to_df(output_path, dim, df_new, row, suffix=SUFFIX_AUG_m30)
                    video_angle_30, df_new = _create_VideoWriter_and_add_to_df(output_path, dim, df_new, row, suffix=SUFFIX_AUG_30)
                    video_zoom_2, df_new = _create_VideoWriter_and_add_to_df(output_path, dim, df_new, row, suffix=SUFFIX_AUG_zoom_2)

                # create the list of frames to iterate
                frames_list = [frame for frame in range(row['WindowOnsetFrame'], row['WindowOffsetFrame'] + 1)]

                # pad shorter videos with the last frame duplicated
                if pad_frames:
                    pad_number = (MINIMUM_NUMBER_OF_FRAMES - 1) - len(frames_list)
                    frames_list = frames_list + [frames_list[-1]]*pad_number
                    if len(frames_list) != MINIMUM_NUMBER_OF_FRAMES-1:
                        print("len(frames_list)", len(frames_list), "Not equal MINIMUM_NUMBER_OF_FRAMES-1", MINIMUM_NUMBER_OF_FRAMES-1, "in index:", index)
                        return
            else:
                # if we dont wan't videos, we'll take only first frame
                frames_list = [row['WindowOnsetFrame']]

            # number of leadizg zeros in SAMM is changing (sometime 0001 or 000001)
            samm_zeros_padding_number=None
            if _is_samm(row['Subject']):
                samm_zeros_padding_number = len(os.listdir(row['MEAbsolutePath'])[0].replace('.', '_').split('_')[1])  # taking length of  0123 out of 009_0123.jpg

            # flags to do stuff only at the first iteration
            need_to_save_first_frame = create_stills
            need_to_save_lbp_histogram = create_lbp
            face = None

            # iterating frames in window
            print("[INFO] Loading: ", row['MEAbsolutePath'])
            for frame in frames_list:
                # set file name
                file_name = _get_filename_from_subject_and_dataset_name(row, frame, samm_zeros_padding_number)

                # check the frame exists in dataset original path
                path_to_load = row['MEAbsolutePath'] + file_name
                if not os.path.exists(path_to_load):  # verify frame exists in source
                    print("Couldn't find: ", path_to_load)
                    return

                # read image
                img = cv2.imread(path_to_load)  # Load frame

                # detect face in the first frame only
                if face is None:
                    face = detect_face(img)

                # crop face
                img = crop_face(img, face)

                # resize image
                img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

                # save first frame if needed
                if need_to_save_first_frame:
                    first_frame_output_path = row['SlowFastSubjectFirstFramesRelativePath'] + row['FileName'] + '.jpg'
                    print(f"[INFO] Saving First Frame to {first_frame_output_path}")
                    cv2.imwrite(first_frame_output_path, img)

                    # if no augmentation will be made, the flag will be set as False, otherwise True and later be set False
                    need_to_save_first_frame = True if augment else False

                # save lbp frame if needed
                if need_to_save_lbp_histogram:
                    lbp_histogram = get_lgbp_histogram(img)
                    lbp_output_path = row['SlowFastSubjectLbpRelativePath'] + row['FileName'] + '.csv'
                    # pd.DataFrame(lbp_histogram).to_csv(lbp_output_path, header=None, index=None)
                    print(f"[INFO] Saving LGBPHS to {lbp_output_path}")
                    np.savetxt(lbp_output_path, lbp_histogram, delimiter=',')

                    # if no augmentation will be made, the flag will be set as False, otherwise True and later be set False
                    need_to_save_lbp_histogram = True if augment else False


                # write the frame to the video already saved
                if create_videos:
                    video.write(img)  # write to video

                # create the augmented data files
                if augment:
                    img_angle_m30 = aug.rotate(img, -30, dim, zoom_factor=1.35)
                    img_angle_30 = aug.rotate(img, 30, dim, zoom_factor=1.35)
                    img_zoom_2 = aug.zoom(img, dim, zoom_factor=2)

                    # write to video
                    if create_videos:
                        video_angle_m30.write(img_angle_m30)
                        video_angle_30.write(img_angle_30)
                        video_zoom_2.write(img_zoom_2)

                    # create augmented data first frames files, if needed
                    if need_to_save_first_frame:
                        first_frame_output_path = row['SlowFastSubjectFirstFramesRelativePath'] + row['FileName'] + '.jpg'
                        cv2.imwrite(first_frame_output_path + SUFFIX_AUG_m30, img_angle_m30)
                        cv2.imwrite(first_frame_output_path + SUFFIX_AUG_30, img_angle_30)
                        cv2.imwrite(first_frame_output_path + SUFFIX_AUG_zoom_2, img_zoom_2)
                        need_to_save_first_frame = False

                    # create augmented data lbp frames files, if needed
                    if need_to_save_lbp_histogram:
                        # lbp_histogram_angle_m30 = lbp_histogram(img_angle_m30)
                        # lbp_histogram_angle_30 = lbp_histogram(img_angle_30)
                        # lbp_histogram_zoom_2 = lbp_histogram(img_zoom_2)
                        #
                        # lbp_output_path = row['SlowFastSubjectLbpRelativePath'] + row['FileName'] + '.csv'
                        #
                        # np.savetxt(lbp_output_path + SUFFIX_AUG_m30, lbp_histogram_angle_m30, delimiter=',')
                        # np.savetxt(lbp_output_path + SUFFIX_AUG_30, lbp_histogram_angle_30, delimiter=',')
                        # np.savetxt(lbp_output_path + SUFFIX_AUG_zoom_2, lbp_histogram_zoom_2, delimiter=',')

                        need_to_save_lbp_histogram = False

            # finish video
            if create_videos:
                cv2.destroyAllWindows()
                video.release()
                if augment:
                    video_angle_m30.release()
                    video_angle_30.release()
                    video_zoom_2.release()

            # cap = cv2.VideoCapture(output_path + '.mp4')
            # print("Number of frames is: ", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            # cap.release()

    return df_new


def split_train_test_and_create_csv(
        df,
        test_size=0.5,
        val_size=0.15,
        random_state=42,
        csv_directory_path=SLOWFAST_DATA_PATH,
        csv_copy_path=""
):
    df['CsvPath'] = SLOWFAST_DATA_PATH + df['Subject'] + "/" + df['FileName'] + ".mp4"
    with open(CLASSIDS_JSON_PATH) as json_file:
        class_mapping = json.load(json_file)
    df = df.replace({'Subject': class_mapping})

    v = df[['Subject']]
    # df = df[v.replace(v.stack().value_counts()).gt(1).all(1)]

    X = df[['CsvPath', 'Subject']]
    y = df['Subject']  # don't need. only for next functions (they need a y)

    # splitting to train + validation and test
    X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # splitting train + validation to test and validation separate
    train_size = 1-test_size
    validation_size_adjusted = val_size / train_size
    X_train, X_val, y_train, y_val = train_test_split(X_train_and_val, y_train_and_val, test_size =validation_size_adjusted, random_state=random_state)

    print(f"number of rows in train ({1-test_size-val_size}) is: {len(X_train)} which is {round((len(X_train)/len(X)), 2)}.")
    print(f"number of rows in val ({val_size}) is: {len(X_val)} which is {round((len(X_val)/len(X)), 2)}")
    print(f"number of rows in test ({test_size}) is: {len(X_test)} which is {round((len(X_test)/len(X)), 2)}")

    print(f"train is missing this labels: \t{sorted(set(y)-set(y_train))}")
    print(f"valid is missing this labels: \t{sorted(set(y)-set(y_val))}")
    print(f"train_and_val is missing this labels: \t{sorted(set(y)-set(y_train_and_val))}")
    print(f"test is missing this labels: \t{sorted(set(y)-set(y_test))}")

    # print(X_train.head(2))
    # print(X_val.head(2))
    # print(X_test.head(2))

    # saving csv
    X_train.to_csv(csv_directory_path + "train.csv", index=False, header=False)
    X_val.to_csv(csv_directory_path + "val.csv", index=False, header=False)
    X_test.to_csv(csv_directory_path + "test.csv", index=False, header=False)

    # saving copys in general outputs
    X_train.to_csv(os.path.join(csv_copy_path, "train.csv"), index=False, header=False)
    X_val.to_csv(os.path.join(csv_copy_path, "val.csv"), index=False, header=False)
    X_test.to_csv(os.path.join(csv_copy_path, "test.csv"), index=False, header=False)
    return


# incomplete
def make_dataset_copy_only_au_frames(df, src_path, dest_path):
    if _is_casme2(src_path):
        parent_dest_directory = dest_path + 'CASME2/'
    elif _is_samm(src_path):
        parent_dest_directory = dest_path + 'SAMM/'
    else:
        raise ValueError(UNKNOWN_PATH_ERROR)
    # make dataset ("parent") directory
    if not os.path.exists(parent_dest_directory):
        os.mkdir(parent_dest_directory)
    for row in df:
        subject_dest_path = parent_dest_directory + row['MERelativePath'].str.split('/')[0]
        file_dest_path = parent_dest_directory + row['MERelativePath']
        if not os.path.exists(subject_dest_path):  # make subject directory
            os.mkdir(subject_dest_path)
        if not os.path.exists(file_dest_path):  # make subject's file directory
            os.mkdir(file_dest_path)


# endregion


# region Helper Functions

def check_if_directory_exist(path, create=False):
    if not os.path.exists(path):
        print("Nothing in path: ", path, " going to create a directory")
        if create:
            os.mkdir(path)


def _create_VideoWriter_and_add_to_df(output_path, dim, df=None, row=None, suffix=''):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path + suffix + '.mp4', fourcc, 200, dim)

    if df is None or row is None:
        return video, None
    else:
        df = df.append(row.replace(row["FileName"], row["FileName"] + suffix))
        return video, df


def _get_filename_from_subject_and_dataset_name(row, frame, samm_zeros_padding_number):
    if _is_casme2(row['Subject']):
        file_name = "img" + str(frame) + '.jpg'  # for example: img1.jpg
    elif _is_samm(row['Subject']):
        file_name = str(row['SubjectNumber']).zfill(3) + '_' + str(frame).zfill(samm_zeros_padding_number) + '.jpg'  # for example: 009_0123.jpg
    elif _is_smic(row['Subject']):
        file_name = "image" + str(frame).zfill(6) + '.jpg'  # for example: image835057.jpg
        pass
    return file_name


def _is_samm(path):
    return "SAMM" in path


def _is_casme2(path):
    return "CASME2" in path

def _is_smic(path):
    return "SMIC" in path


def _filter_selected_subjects(df, selected_subjects):
    return df[df['Subject'].isin(selected_subjects)]


def _remove_rows_under_minimum_frames_number(df):
    return df[df['NumberOfFrames'] > MINIMUM_NUMBER_OF_FRAMES]


def _rename_subjects_name(df, path):
    if _is_samm(path):
        dataset_name = "SAMM_"
    elif _is_casme2(path):
        dataset_name = "CASME2_"
    elif _is_smic(path):
        dataset_name = "SMIC_"
    df['SubjectNumber'] = df['Subject']
    df['Subject'] = dataset_name + df['Subject'].astype(str)
    return df


def _get_accurate_emotions_in_smic(df, path):
    if _is_smic(path):
        df.loc[df.FileName.str.contains('ne'), 'Emotion'] = "negative"
        df.loc[df.FileName.str.contains('po'), 'Emotion'] = "positive"
        df.loc[df.FileName.str.contains('sur'), 'Emotion'] = "surprise"
    return df

def _remove_duplicate_columns(df, path):
    if _is_smic(path):
        df.pop('Onset2F')
        df.pop('Offset2F')
        df.pop('Onset3F')
        df.pop('Offset3F')
        df.pop('TotalMF2')
        df.pop('TotalMF3')
    return df


def _remove_empty_columns(df, path):
    if _is_casme2(path):
        df.pop('Unnamed: 2')
        df.pop('Unnamed: 6')
    if _is_smic(path):
        df.pop('Unnamed: 2')
        df.pop('Unnamed: 9')
        df.pop('Unnamed: 17')
        df.pop('Unnamed: 18')

    return df


def _rename_columns_to_pascal(df, path):
    if _is_casme2(path):
        naming_dict = {
            'Filename': 'FileName',
            'Action Units': 'ActionUnits',
            'Estimated Emotion': 'EstimatedEmotion'
        }
    elif _is_samm(path):
        naming_dict = {
            'Filename': 'FileName',
            'Inducement Code': 'InducementCode',
            'Onset Frame': 'OnsetFrame',
            'Apex Frame': 'ApexFrame',
            'Offset Frame': 'OffsetFrame',
            'Action Units': 'ActionUnits',
            'Estimated Emotion': 'EstimatedEmotion',
            'Objective Classes': 'ObjectiveClasses',
            'Duration': 'NumberOfFrames'
        }
    elif _is_smic(path):
        naming_dict = {
            'Filename': 'FileName',
            'OnsetF': 'OnsetFrame',
            'OffsetF': 'OffsetFrame',
            'FirstF': 'FirstFrame',
            'LastF': 'LastFrame',
            'TotalMF1': 'NumberOfFrames',
            'TotalVL': 'TotalVL'
        }
    else:
        raise ValueError(UNKNOWN_PATH_ERROR)

    df.rename(columns=naming_dict, inplace=True)
    return df


def _add_apex_frame_column_for_smic(df, path):
    if _is_smic(path):
        df['ApexFrame'] = (df['OffsetFrame'] + df['OnsetFrame']) // 2
        # df['Apex2Frame'] = (df['Offset2Frame'] + df['Onset2Frame']) // 2
        # df['Apex3Frame'] = (df['Offset3Frame'] + df['Onset3Frame']) // 2
    return df

def _convert_string_series_to_int(df):
    df['OffsetFrame'] = df['OffsetFrame'].astype(int)
    df['OnsetFrame'] = df['OnsetFrame'].astype(int)
    df['ApexFrame'] = df['ApexFrame'].astype(int)
    return df


def _add_columns_from_other_df(df, path):
    if _is_casme2(path):
        df['NumberOfFrames'] = df['OffsetFrame'] - df[
            'OnsetFrame'] + 1  # SAMM doesn't have 'Duration' column but should have
        df['InducementCode'] = np.nan
        df['Micro'] = np.nan
        df['ObjectiveClasses'] = np.nan
        df['Notes'] = np.nan
    return df


def _drop_problematic_rows(df, path):
    if _is_casme2(path):
        df = df.drop([29])  # row 29 in CASME2 has the value '/' in ApexFrame
    if _is_samm(path):
        df = df.drop([132, 125])  # Apex is smaller then Onset or bigger the Offset
    return df


def _get_file_path(df, path, is_absolute, is_subject):
    file_path = ''
    if _is_samm(path):  # for example: ../../../../data/galsha/SAMM/SAMM/006/006_1_2/
        if is_absolute:
            file_path = SAMM_DATASET_PATH
        file_path = file_path + df['SubjectNumber'].astype(str).str.zfill(3) + '/'
        if not is_subject:
            file_path = file_path + df['FileName'] + '/'
    elif _is_casme2(path):  # for example: ../../../../data/galsha/CASME2/CASME2-RAW/sub01/EP02_01f/
        if is_absolute:
            file_path = CASME2_DATASET_PATH
        file_path = file_path + "sub" + df['SubjectNumber'].astype(str).str.zfill(2) + '/'
        if not is_subject:
            file_path = file_path + df['FileName'] + "/"
    elif _is_smic(path):
        if is_absolute:
            file_path = SMIC_DATASET_PATH
        file_path = file_path + "s" + df['SubjectNumber'].astype(str).str.zfill(2) + '/'
        if not is_subject:
            file_path = file_path + df['FileName'] + "/"


    else:
        raise ValueError(UNKNOWN_PATH_ERROR)
    return file_path


def _add_paths_columns(df, path):
    # Dataset paths - Subject
    df['SubjectAbsolutePath'] = _get_file_path(df, path, is_absolute=True, is_subject=True)
    df['SubjectRelativePath'] = _get_file_path(df, path, is_absolute=False, is_subject=True)

    # Dataset paths - ME
    df['MEAbsolutePath'] = _get_file_path(df, path, is_absolute=True, is_subject=False)
    df['MERelativePath'] = _get_file_path(df, path, is_absolute=False, is_subject=False)

    # SlowFast videos paths
    df['SlowFastSubjectRelativePath'] = SLOWFAST_DATA_PATH + df['Subject'] + "/"

    # First frames paths
    df['SlowFastSubjectFirstFramesRelativePath'] = SLOWFAST_FIRST_FRAMES_DATA_PATH + df['Subject'] + "/"

    # LBP paths
    df['SlowFastSubjectLbpRelativePath'] = SLOWFAST_LBP_DATA_PATH + df['Subject'] + "/"
    return df


def _reset_index_if_needed(df, reset_index):
    if reset_index:
        df = df.reset_index()
    return df


def _explode_action_units_combinations(df, split_multiple_au, remove_letters):
    if split_multiple_au:
        df.ActionUnits = df.ActionUnits.str.replace('or', '+').str.split('+').tolist()
        df = df.explode('ActionUnits')
        if remove_letters:
            df['ActionUnits'] = df['ActionUnits'].str.extract('(\d+)', expand=False)
    return df


def _calculate_frames_window_from_apex(df, path):
    df['OffsetDifferenceFromApexFrame'] = df['OffsetFrame'] - df['ApexFrame']
    df['OnsetDifferenceFromApexFrame'] = df['ApexFrame'] - df['OnsetFrame']
    df['MaximumWindowFramesFromApex'] = df[['OffsetDifferenceFromApexFrame', 'OnsetDifferenceFromApexFrame']].min(
        axis=1) * 2 + 1

    # Creating window
    half_window = (WINDOW_FROM_APEX_MIN_FRAMES) / 2  # UNIFIED_MIN_FRAMES is an odd number because ApexFrame is the center
    left_half = half_window
    right_half = half_window-1
    df['WindowOnsetFrame'] = (df['ApexFrame'] - left_half).astype('int')
    df['WindowOffsetFrame'] = (df['ApexFrame'] + right_half).astype('int')
    for index, row in df.iterrows():
        # print("index", index, row)
        if row['NumberOfFrames'] < MINIMUM_NUMBER_OF_FRAMES:  # video is shorter than minimum frames per final video
            df.loc[index, 'WindowOffsetFrame'] = df.loc[index, 'OffsetFrame']
            df.loc[index, 'WindowOnsetFrame'] = df.loc[index, 'OnsetFrame']
        elif row['WindowOnsetFrame'] < row['OnsetFrame']:  # Start of window is before first frame
            df.loc[index, 'WindowOffsetFrame'] += (
                        row['OnsetFrame'] - row['WindowOnsetFrame'])  # Add extra to the end of the window
            df.loc[index, 'WindowOnsetFrame'] = row['OnsetFrame']  # Move start of window to first frame
        elif row['OffsetFrame'] < row['WindowOffsetFrame']:  # End of window is after last frame
            df.loc[index, 'WindowOnsetFrame'] -= (
                        row['WindowOffsetFrame'] - row['OffsetFrame'])  # Add extra to the start of the window
            df.loc[index, 'WindowOffsetFrame'] = row['OffsetFrame']  # Move end of window to last frame
        # print("index", index, row)

    # Verify that indeed all window frames are in the actual video
    df['StartBeforeOnset'] = df['WindowOnsetFrame'] < df['OnsetFrame']
    df['EndAfterOffset'] = df['OffsetFrame'] < df['WindowOffsetFrame']
    df['WindowOnsetFrameToOffsetFrame'] = df['WindowOffsetFrame'] - df['WindowOnsetFrame'] + 1

    return df


def _find_missing(lst, Onset, Offset):
    return [x for x in range(int(Onset), int(Offset) + 1) if x not in lst]


# endregion


