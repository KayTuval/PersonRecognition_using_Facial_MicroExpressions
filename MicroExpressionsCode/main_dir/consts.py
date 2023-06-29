import os

# region Global Variables

# Paths
DATASETS_PATH = "/mnt/storage/datasets/MEDatasets/"

SAMM_DIR_PATH = os.path.join(DATASETS_PATH, "SAMM/")
SAMM_CSV_PATH = os.path.join(SAMM_DIR_PATH, "SAMM/SAMM_Micro_FACS_Codes_v2.csv")
SAMM_DATASET_PATH = os.path.join(SAMM_DIR_PATH, "SAMM/")

CASME2_DIR_PATH = os.path.join(DATASETS_PATH, "CASME2/")
CASME2_CSV_PATH = os.path.join(CASME2_DIR_PATH, "CASME2-coding-20190701.csv")
CASME2_DATASET_PATH = os.path.join(CASME2_DIR_PATH, "CASME2-RAW/")

SMIC_DIR_PATH = os.path.join(DATASETS_PATH, "SMIC/")
SMIC_CSV_PATH = os.path.join(SMIC_DIR_PATH, "SMIC_E_raw_image/HS_long/SMIC-HS-E/SMIC-HS-E_annotation.csv")
SMIC_DATASET_PATH = os.path.join(SMIC_DIR_PATH, "SMIC_E_raw_image/HS_long/SMIC-HS-E/")

PROJECT_PATH = '/home/khen_proj_1/'

SLOWFAST_DATA_PATH = os.path.join(PROJECT_PATH, "PycharmProjects/slowfast_test/slowfast/data/MicroExpressions/")
CLASSIDS_JSON_PATH = os.path.join(SLOWFAST_DATA_PATH, 'classids.json')
CLASSES_TXT_PATH = os.path.join(SLOWFAST_DATA_PATH, 'classes.txt')

SLOWFAST_FIRST_FRAMES_DATA_PATH = os.path.join(PROJECT_PATH, "PycharmProjects/slowfast_test/slowfast/data/MicroExpressionsFirstFrames/")
SLOWFAST_LBP_DATA_PATH = os.path.join(PROJECT_PATH, "PycharmProjects/slowfast_test/slowfast/data/MicroExpressionsLBP/")

SLOWFAST_TRAIN_PATH = os.path.join(SLOWFAST_DATA_PATH, "train.csv")
SLOWFAST_VAL_PATH = os.path.join(SLOWFAST_DATA_PATH, "val.csv")
SLOWFAST_TEST_PATH = os.path.join(SLOWFAST_DATA_PATH, "test.csv")

SVM_OUTPUT_PATH = '/home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/SVM'
CLASSES_OUTPUT_PATH = '/home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/classes'

UNKNOWN_PATH_ERROR = "Unknown path. Choose between CASME2, SAMM or SMIC"

SLOWFAST_RUN_NET_PATH = '/home/khen_proj_1/PycharmProjects/slowfast_test/slowfast/tools/run_net.py'


# FPS
SAMM_FPS = 200
CASME2_FPS = 200
SMIC_FPS = 100

# Spatial
SAMM_WIDTH = 960
SAMM_HEIGHT = 650

CASME2_WIDTH = 640
CASME2_HEIGHT = 480

SMIC_WIDTH = 640
SMIC_HEIGHT = 480

# Resized Spatials
SAMM_RESIZED_WIDTH = 400
SAMM_RESIZED_HEIGHT = 400

CASME2_RESIZED_WIDTH = 300
CASME2_RESIZED_HEIGHT = 300

SMIC_RESIZED_WIDTH = 150
SMIC_RESIZED_HEIGHT = 150

REDUCED_WIDTH = 256
REDUCED_HEIGHT = 256

# Frames Information
CASME2_MIN_FRAMES = 24  # minimum frames per video in CASME2 is 24, maximum is 141
SAMM_MIN_FRAMES = 30  # minimum frames per video in SAMM is 30, maximum is 101
SMIC_MIN_FRAMES = 16  # minimum frames per video in SAMM is 16, maximum is 58
MINIMUM_NUMBER_OF_FRAMES = 65  # the number should the number requested plus 1. if we want 64, write 65.
WINDOW_FROM_APEX_MIN_FRAMES = MINIMUM_NUMBER_OF_FRAMES - 1  # Minus 1 to be an odd number # min(CASME2_MIN_FRAMES, SAMM_MIN_FRAMES) - 1
# WINDOW_FROM_APEX_MIN_FRAMES = min(CASME2_MIN_FRAMES, SAMM_MIN_FRAMES) - 1  # Minus 1 to be an odd number #

# Dataframes Parameters
SELECTED_SUBJECTS = [
    "CASME2_17",
    "SAMM_11",
    "CASME2_5",
    "CASME2_26",
    "CASME2_19"
]

GENERAL_DF_PARAMS = {
    "remove_rows_under_minimum_frames_number": False,
    "selected_subjects": [],  # SELECTED_SUBJECTS,
    "split_multiple_au": False,
    "remove_letters": False,
    "reset_index": False,
    "validate_files_exists": True
}
SAMM_DF_ARGS = {
    "path": SAMM_CSV_PATH,
    **GENERAL_DF_PARAMS
}
CASME2_DF_ARGS = {
    "path": CASME2_CSV_PATH,
    **GENERAL_DF_PARAMS
}

SMIC_DF_ARGS = {
    "path": SMIC_CSV_PATH,
    **GENERAL_DF_PARAMS
}

UNIONED_DF_ARGS = {
    "CASME2": CASME2_DF_ARGS,
    "SAMM": SAMM_DF_ARGS,
    "SMIC": SMIC_DF_ARGS
}

# Frames Name (SAMM and CASME2)
FRAMES_REGEX_PATTERN = "(?:(?<=_)|(?<=img)|(?<=image))(\d+)"

# Augmentation Suffices
SUFFIX_AUG_30 = "_angle_30"
SUFFIX_AUG_m30 = "_angle_m30"
SUFFIX_AUG_zoom_2 = "_zoom_2"

# endregion
