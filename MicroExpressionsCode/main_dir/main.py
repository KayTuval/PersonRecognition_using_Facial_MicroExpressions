import datetime

import pandas as pd
import sys
import argparse

sys.path.insert(0, '/home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/')
from datasets.DatasetsUtils import *
from main_dir.consts import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset arguments
    parser.add_argument('--data', type=str, required=True,
                        help='SAMM, CASME2 or SMIC')

    # Create data arguments
    parser.add_argument('--pad_frames', default=False, type=bool,
                        help='If window around apex of the ME is shorter than the minimum frame length, it will pad the video with extra frames')
    parser.add_argument('--create_videos', default=False, type=bool, required='--pad_frames' in sys.argv,
                        help='Merge frames to get video files')
    parser.add_argument('--create_stills', default=False, type=bool,
                        help='Save first frames')
    parser.add_argument('--create_lbp', default=False, type=bool,
                        help='Create lbp vector')
    parser.add_argument('--augment', default=False, type=bool,
                        help='Use data augmentations')
    parser.add_argument('--create_classesids_files', default=True, type=bool,
                        help='Create classesids.json and classes.txt')
    parser.add_argument('--min_samples', default=2, type=int,
                        help='drop every subject that has less than this number of samples')

    # path arguments
    parser.add_argument('--copy_path', default=CLASSES_OUTPUT_PATH, type=str,
                        help='making a copy of json, txt or csv files')
    parser.add_argument('--copy_path_folder_name', default=str(str(datetime.datetime.now())), type=str,
                        help='Name of folder in copy_path')

    # split train test argumetns
    parser.add_argument('--split_data', default=False, type=bool,
                        help='Split data into train, validation and test. save it in csv files')
    parser.add_argument('--val_size', default=0.1, type=int,
                        help='Size of validation set')
    parser.add_argument('--test_size', default=0.5, type=int,
                        help='Size of Test set')
    parser.add_argument('--random_state', default=42, type=int,
                        help='Random State of split')
    parser.add_argument('--csv_directory_path', default=SLOWFAST_DATA_PATH, type=str,
                        help='Path to location of where to save csv files')

    # Pandas arguments
    parser.add_argument('--show_df', default=True, type=bool,
                        help='Whether to print df')
    parser.add_argument('--show_max_df_data', default=True, type=bool,
                        help='Whether to truncate the data presented')

    # parse arguments
    args = parser.parse_args()

    # apply arguments
    if args.show_max_df_data:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 100)

    # Join classesids output path and output folder
    copy_path_full = os.path.join(args.copy_path, args.copy_path_folder_name, args.data)

    # create parameters for data making
    create_data_params = {
        'create_videos': args.create_videos,
        'augment': args.augment,
        'pad_frames': args.pad_frames,
        'create_stills': args.create_stills,
        'create_lbp': args.create_lbp,
        'create_classesids_files': args.create_classesids_files,
        'copy_path': copy_path_full
    }

    split_data_params = {
        'test_size': args.test_size,
        'val_size': args.val_size,
        'random_state': args.random_state,
        'csv_directory_path': args.csv_directory_path,
        'csv_copy_path': copy_path_full
    }

    # run functions
    # create df
    if args.data == "SAMM":
        df = get_df(**SAMM_DF_ARGS, min_samples=args.min_samples)
    elif args.data == "CASME2":
        df = get_df(**CASME2_DF_ARGS, min_samples=args.min_samples)
    elif args.data == "SMIC":
        df = get_df(**SMIC_DF_ARGS, min_samples=args.min_samples)

    # show df
    if args.show_df:
        df.head(10)

    # make data
    if args.create_videos or args.create_stills or args.create_lbp or args.create_classesids_files:
        df = make_videos_of_microexpression_in_slowfast_directory(df, **create_data_params)

    print("number of classes after creating the data is: ", df["Subject"].nunique())

    # split data
    if args.split_data:
        split_train_test_and_create_csv(df, **split_data_params)



# path = '/home/khen_proj_1/PycharmProjects/slowfast_test/slowfast/data/MicroExpressionsFirstFrames/CASME2_11/EP13_03f.jpg'
# img = cv2.imread(path, 0)
# img_lbp = get_lgbp_histogram(img)

# classifier(dataset="SAMM", lbp=True, min_samples=1)


# region Notes
""" 

classifier(dataset="SAMM", min_samples=1)  # Accuracy Score: 0.7088607594936709


minimum frames per video in CASME2 is 24, maximum is 141
minimum frames per video in SAMM is 30, maximum is 101


Subjects with the most videos:
CASME2_17    36
SAMM_11      20
CASME2_5     19
CASME2_26    17
CASME2_19    16
CASME2_10    14
CASME2_9     14
SAMM_26      13
CASME2_2     13
CASME2_23    12
CASME2_12    12
CASME2_20    11
SAMM_6       11
SAMM_14      11
SAMM_7       10

Subject ranking for files over 40 frames
CASME2_17    31
SAMM_11      19
CASME2_5     19
CASME2_26    16
CASME2_19    15
CASME2_10    14
CASME2_9     14
CASME2_2     13
SAMM_26      12
CASME2_23    12

Subject ranking for files over 64 frames
CASME2_17    17
SAMM_11      12
SAMM_26      11
SAMM_6       10
CASME2_12    10
CASME2_9     10
CASME2_23     9
CASME2_10     9
CASME2_2      9
SAMM_14       8


In CASME2 and SAMM together, The number of frames above 70 is: 206
In CASME2 and SAMM together, The number of frames above 60 is: 262
In CASME2 and SAMM together, The number of frames above 50 is: 328
In CASME2 and SAMM together, The number of frames above 40 is: 380
In CASME2 and SAMM together, The number of frames above 30 is: 408
In CASME2 and SAMM together, The number of frames above 20 is: 411

"""
# endregion

# region Running Commands
"""

python slowfast\tools\run_net.py --cfg slowfast/configs/ME/SLOWFAST_8x8_R50_stepwise_multigrid.yaml


"""
# endregion
