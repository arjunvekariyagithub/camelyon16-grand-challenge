import getpass

users = ['arjun', 'millpc']

user = getpass.getuser()
print('user: %s' % user)

assert user in users, 'User not Authorised!!'

data_subset = ['train', 'train-aug', 'validation', 'validation-aug', 'heatmap']

DATA_DIR = '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/'

TUMOR_WSI_PATH = DATA_DIR + 'TrainingData/Train_Tumor'
NORMAL_WSI_PATH = DATA_DIR + 'TrainingData/Train_Normal'
TUMOR_MASK_PATH = DATA_DIR + 'TrainingData/Ground_Truth/Mask'

PATCHES_TRAIN_DIR = DATA_DIR + 'Processed/patch-based-classification/raw-data/train/'
PATCHES_VALIDATION_DIR = DATA_DIR + 'Processed/patch-based-classification/raw-data/validation/'
PATCHES_TRAIN_NEGATIVE_PATH = PATCHES_TRAIN_DIR + 'label-0/'
PATCHES_TRAIN_POSITIVE_PATH = PATCHES_TRAIN_DIR + 'label-1/'
PATCHES_VALIDATION_NEGATIVE_PATH = PATCHES_VALIDATION_DIR + 'label-0/'
PATCHES_VALIDATION_POSITIVE_PATH = PATCHES_VALIDATION_DIR + 'label-1/'

PATCHES_TRAIN_AUG_DIR = DATA_DIR + 'Processed/patch-based-classification/raw-data-aug/train/'
PATCHES_VALIDATION_AUG_DIR = DATA_DIR + 'Processed/patch-based-classification/raw-data-aug/validation/'
PATCHES_TRAIN_AUG_NEGATIVE_PATH = PATCHES_TRAIN_AUG_DIR + 'label-0/'
PATCHES_TRAIN_AUG_POSITIVE_PATH = PATCHES_TRAIN_AUG_DIR + 'label-1/'
PATCHES_TRAIN_AUG_EXCLUDE_MIRROR_WSI_NEGATIVE_PATH = PATCHES_TRAIN_AUG_DIR + 'exclude-mirror-label-0/'
PATCHES_TRAIN_AUG_EXCLUDE_MIRROR_WSI_POSITIVE_PATH = PATCHES_TRAIN_AUG_DIR + 'exclude-mirror-label-1/'
PATCHES_VALIDATION_AUG_NEGATIVE_PATH = PATCHES_VALIDATION_AUG_DIR + 'label-0/'
PATCHES_VALIDATION_AUG_POSITIVE_PATH = PATCHES_VALIDATION_AUG_DIR + 'label-1/'

TRAIN_TF_RECORDS_DIR_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/tf-records/',
    'millpc': DATA_DIR + 'Processed/patch-based-classification/tf-records/'
}

HEAT_MAP_RAW_PATCHES_DIR_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/heat-map/patches/raw/',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/Processed/heat-map/patches/raw/'
}

HEAT_MAP_TF_RECORDS_DIR_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/heat-map/patches/tf-records/',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/Processed/'
              'heat-map/patches/tf-records/'
}

HEAT_MAP_WSIs_PATH_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/heat-map/WSIs/',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/Processed/heat-map/WSIs/'
}

HEAT_MAP_DIR_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/heat-map/heatmaps/',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/Processed/heat-map/heatmaps/'
}

FINE_TUNE_MODEL_CKPT_PATH_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/training/model3/model.ckpt-60000',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/training/successful_model/model3/model.ckpt-60000'
}

EVAL_MODEL_CKPT_PATH_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/training/model6/model.ckpt-300000',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/training/all_models/model6/model.ckpt-300000'
}

TRAIN_DIR_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/training/model7/',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/training/all_models/model7/'
}

EVAL_DIR_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/evaluation',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/evaluation'
}

TRAIN_TF_RECORDS_DIR = TRAIN_TF_RECORDS_DIR_LIST[user]
TRAIN_DIR = TRAIN_DIR_LIST[user]
EVAL_DIR = EVAL_DIR_LIST[user]
HEAT_MAP_RAW_PATCHES_DIR = HEAT_MAP_RAW_PATCHES_DIR_LIST[user]
HEAT_MAP_TF_RECORDS_DIR = HEAT_MAP_TF_RECORDS_DIR_LIST[user]
HEAT_MAP_WSIs_PATH = HEAT_MAP_WSIs_PATH_LIST[user]
HEAT_MAP_DIR = HEAT_MAP_DIR_LIST[user]
FINE_TUNE_MODEL_CKPT_PATH = FINE_TUNE_MODEL_CKPT_PATH_LIST[user]
EVAL_MODEL_CKPT_PATH = EVAL_MODEL_CKPT_PATH_LIST[user]

PATCH_SIZE = 256
PATCH_NORMAL_PREFIX = 'normal_'
PATCH_TUMOR_PREFIX = 'tumor_'
PATCH_AUG_NORMAL_PREFIX = 'aug_false_normal_'
PATCH_AUG_TUMOR_PREFIX = 'aug_false_tumor_'
PREFIX_SHARD_TRAIN = 'train'
PREFIX_SHARD_AUG_TRAIN = 'train-aug'
PREFIX_SHARD_VALIDATION = 'validation'
PREFIX_SHARD_AUG_VALIDATION = 'validation-aug'

BATCH_SIZE = 32

NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX = 100
NUM_POSITIVE_PATCHES_FROM_EACH_BBOX = 500
PATCH_INDEX_NEGATIVE = 700000
PATCH_INDEX_POSITIVE = 700000

TUMOR_PROB_THRESHOLD = 0.90
PIXEL_WHITE = 255
PIXEL_BLACK = 0


def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    return filename


def step_range(start, end, step):
    while start <= end:
        yield start
        start += step
