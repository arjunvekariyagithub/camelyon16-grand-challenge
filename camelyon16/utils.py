import getpass

users = ['arjun', 'millpc']

user = getpass.getuser()

assert user in users, 'User not Authorised!!'

DATA_DIR = '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/'

TUMOR_WSI_PATH = DATA_DIR + 'TrainingData/Train_Tumor'
NORMAL_WSI_PATH = DATA_DIR + 'TrainingData/Train_Normal'
TUMOR_MASK_PATH = DATA_DIR + 'TrainingData/Ground_Truth/Mask'

PATCHES_TRAIN_NEGATIVE_PATH = DATA_DIR + 'Processed/patch-based-classification/raw-data/train/label-0/'
PATCHES_TRAIN_POSITIVE_PATH = DATA_DIR + 'Processed/patch-based-classification/raw-data/train/label-1/'
PATCHES_VALIDATION_NEGATIVE_PATH = DATA_DIR + \
                                             'Processed/patch-based-classification/raw-data/validation/label-0/'
PATCHES_VALIDATION_POSITIVE_PATH = DATA_DIR + \
                                             'Processed/patch-based-classification/raw-data/validation/label-1/'

PATCHES_TRAIN_AUG_NEGATIVE_PATH = DATA_DIR + 'Processed/patch-based-classification/raw-data-aug/train/label-0/'
PATCHES_TRAIN_AUG_POSITIVE_PATH = DATA_DIR + 'Processed/patch-based-classification/raw-data-aug/train/label-1/'
PATCHES_VALIDATION_AUG_NEGATIVE_PATH = DATA_DIR + \
                                             'Processed/patch-based-classification/raw-data-aug/validation/label-0/'
PATCHES_VALIDATION_AUG_POSITIVE_PATH = DATA_DIR + \
                                             'Processed/patch-based-classification/raw-data-aug/validation/label-1/'


TF_RECORDS_PATH = DATA_DIR + 'Processed/patch-based-classification/tf-records/'

HEAT_MAP_RAW_PATCHES_DIR_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/heat-map/patches/raw',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/Processed/heat-map/patches/raw'
}

HEAT_MAP_TF_RECORDS_DIR_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/heat-map/patches/tf-records',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/Processed/'
              'heat-map/patches/tf-records'
}

HEAT_MAP_WSIs_PATH_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/heat-map/WSIs',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/Processed/heat-map/WSIs'
}

HEAT_MAP_DIR_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/heat-map/heatmaps',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/Processed/heat-map/heatmaps'
}

MODEL_CKPT_PATH_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/training/model3/model.ckpt-60000',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/training/successful_model/model3/model.ckpt-60000'
}

TRAIN_DIR_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/training/model3',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/training/model3'
}

EVAL_DIR_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/evaluation',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/evaluation'
}

TRAIN_DIR = TRAIN_DIR_LIST[user]
EVAL_DIR = EVAL_DIR_LIST[user]
HEAT_MAP_RAW_PATCHES_DIR = HEAT_MAP_RAW_PATCHES_DIR_LIST[user]
HEAT_MAP_TF_RECORDS_DIR = HEAT_MAP_TF_RECORDS_DIR_LIST[user]
HEAT_MAP_WSIs_PATH = HEAT_MAP_WSIs_PATH_LIST[user]
HEAT_MAP_DIR = HEAT_MAP_DIR_LIST[user]
MODEL_CKPT_PATH = MODEL_CKPT_PATH_LIST[user]

PATCH_SIZE = 256
PATCH_NORMAL_PREFIX = 'normal_'
PATCH_TUMOR_PREFIX = 'tumor_'
PATCH_AUG_NORMAL_PREFIX = 'aug_normal_'
PATCH_AUG_TUMOR_PREFIX = 'aug_tumor_'
BATCH_SIZE = 32


def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    return filename


def step_range(start, end, step):
    while start <= end:
        yield start
        start += step