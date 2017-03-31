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
TEST_WSI_PATH = DATA_DIR + 'Testset'

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
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/training/model5/model.ckpt-95000',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/training/successful_model/model5/model.ckpt-95000'
}

EVAL_MODEL_CKPT_PATH_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/training/model8/model.ckpt-90000',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/training/all_models/model8/model.ckpt-90000'
}

HEATMAP_MODEL_CKPT_PATH_MILLPC = {
    'model5': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/training/successful_model/model5/model.ckpt-95000',
    'model8': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/training/successful_model/model8/model.ckpt-90000'
}

HEATMAP_MODEL_CKPT_PATH_ARJUN = {
    'model5': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/training/model5/model.ckpt-95000',
    'model8': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/training/model8/model.ckpt-90000'
}

TRAIN_DIR_LIST = {
    'arjun': '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/training/model8/',
    'millpc': '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/training/all_models/model8/'
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

heatmap_models = ['model5', 'model8']
FIRST_HEATMAP_MODEL = 'model5'
SECOND_HEATMAP_MODEL = 'model8'
PATCH_SIZE = 256
PATCH_NORMAL_PREFIX = 'normal_'
PATCH_TUMOR_PREFIX = 'tumor_'
PATCH_AUG_NORMAL_PREFIX = 'aug_false_normal_'
PATCH_AUG_TUMOR_PREFIX = 'aug_false_tumor_'
PREFIX_SHARD_TRAIN = 'train'
PREFIX_SHARD_AUG_TRAIN = 'train-aug'
PREFIX_SHARD_VALIDATION = 'validation'
PREFIX_SHARD_AUG_VALIDATION = 'validation-aug'

n_patches_dic = {'Normal_085': 3640, 'Normal_094': 8016, 'Normal_077': 7837, 'Tumor_028': 13315, 'Normal_081': 7413,
                 'Normal_054': 1232, 'Tumor_014': 8767, 'tumor_086': 22118, 'Tumor_067': 4581, 'Normal_019': 166,
                 'Normal_040': 16255, 'tumor_096': 44417, 'Normal_069': 7859, 'Normal_013': 1636, 'Normal_133': 10456,
                 'Normal_100': 7542, 'Normal_078': 4558, 'Normal_064': 14446, 'Tumor_059': 4393, 'Normal_021': 258,
                 'tumor_094': 54862, 'Normal_066': 25134, 'tumor_098': 30078, 'Normal_135': 47222, 'Normal_117': 9487,
                 'Normal_071': 4849, 'Tumor_031': 2516, 'Normal_144': 21735, 'Tumor_064': 4187, 'Tumor_025': 9416,
                 'Tumor_037': 10565, 'Normal_067': 11733, 'Tumor_046': 16516, 'Normal_007': 3356, 'tumor_104': 79316,
                 'Normal_034': 15178, 'Normal_039': 8278, 'Normal_124': 19310, 'tumor_080': 6752, 'Normal_089': 17769,
                 'Normal_079': 9468, 'Normal_086': 5222, 'Tumor_070': 5560, 'Normal_047': 4994, 'Normal_136': 53434,
                 'Tumor_018': 9500, 'Normal_126': 13070, 'Normal_097': 15928, 'Tumor_020': 10133, 'Normal_127': 59073,
                 'Normal_063': 6176, 'Normal_122': 71643, 'Normal_138': 10704, 'Tumor_060': 7216, 'Normal_043': 442,
                 'Normal_139': 15047, 'tumor_102': 33177, 'Normal_109': 45388, 'tumor_081': 32108, 'Normal_018': 2391,
                 'Normal_112': 73571, 'Normal_023': 756, 'Normal_142': 16726, 'tumor_091': 74683, 'Tumor_016': 4600,
                 'Tumor_032': 16944, 'Normal_099': 15469, 'Normal_114': 102924, 'Normal_020': 523, 'Normal_072': 6722,
                 'Normal_036': 5110, 'Tumor_069': 6788, 'tumor_083': 20614, 'tumor_084': 31939, 'Normal_113': 11785,
                 'Normal_151': 11499, 'Tumor_050': 7462, 'Normal_001': 2322, 'Normal_108': 24398, 'Tumor_065': 2586,
                 'tumor_077': 15618, 'Normal_015': 4258, 'Tumor_039': 5785, 'Normal_012': 3695, 'Normal_008': 345,
                 'Normal_134': 5298, 'Normal_026': 5697, 'Tumor_022': 9780, 'Normal_087': 11914, 'Normal_082': 11569,
                 'Tumor_043': 3623, 'Normal_050': 9614, 'tumor_101': 11813, 'Tumor_002': 3111, 'Normal_010': 2227,
                 'Tumor_007': 10220, 'Normal_156': 38825, 'Normal_146': 13326, 'Tumor_033': 7067, 'Normal_073': 7901,
                 'tumor_076': 63630, 'Normal_116': 20541, 'Normal_154': 13737, 'Normal_056': 4354, 'Normal_158': 16324,
                 'Normal_076': 24075, 'Normal_147': 19119, 'Tumor_003': 6656, 'Normal_074': 9064, 'Normal_070': 16119,
                 'Normal_002': 3815, 'Tumor_012': 5219, 'Tumor_057': 4169, 'tumor_093': 37696, 'Normal_092': 4737,
                 'tumor_092': 36460, 'tumor_103': 17425, 'Normal_052': 14410, 'tumor_079': 15194, 'Normal_131': 62170,
                 'Tumor_054': 13094, 'Normal_119': 40271, 'tumor_100': 54502, 'Normal_120': 13982, 'Normal_095': 13067,
                 'Normal_150': 79728, 'Normal_065': 4714, 'Tumor_030': 13544, 'tumor_097': 69742, 'Normal_014': 1819,
                 'tumor_106': 19832, 'tumor_105': 29918, 'Tumor_055': 8493, 'Tumor_051': 12828, 'Normal_051': 3213,
                 'Normal_160': 16898, 'tumor_082': 6802, 'Normal_053': 15862, 'Tumor_013': 4298, 'Normal_088': 14480,
                 'tumor_073': 67738, 'Normal_091': 27564, 'Tumor_006': 15888, 'Normal_035': 1285, 'Tumor_015': 24441,
                 'Tumor_036': 13704, 'Normal_057': 4682, 'Tumor_045': 8976, 'Normal_103': 25128, 'Normal_115': 37442,
                 'Normal_101': 12107, 'tumor_088': 13742, 'Normal_148': 21481, 'Normal_024': 8568, 'Normal_129': 20765,
                 'Normal_096': 10686, 'Tumor_058': 4197, 'tumor_087': 34051, 'Tumor_040': 14738, 'Tumor_048': 11906,
                 'Normal_140': 46404, 'tumor_071': 11867, 'tumor_072': 17953, 'Tumor_011': 6884, 'tumor_075': 18164,
                 'Normal_104': 15924, 'Normal_068': 6605, 'Normal_055': 5478, 'Normal_044': 5212, 'Tumor_052': 13138,
                 'Tumor_038': 1261, 'Tumor_005': 4067, 'Normal_157': 52461, 'Normal_152': 32036, 'Tumor_063': 10365,
                 'Normal_049': 4331, 'Normal_016': 73, 'Normal_145': 21083, 'Tumor_010': 10966, 'Normal_098': 3047,
                 'Tumor_008': 11704, 'Normal_061': 10263, 'Normal_090': 15123, 'Normal_042': 48, 'Normal_111': 27546,
                 'Normal_048': 2923, 'Normal_062': 2279, 'Tumor_044': 11643, 'Tumor_017': 8350, 'Normal_128': 7904,
                 'Normal_149': 19316, 'tumor_107': 37056, 'Normal_125': 79664, 'Normal_130': 52386, 'tumor_089': 26618,
                 'Normal_028': 3471, 'Tumor_049': 3142, 'Normal_118': 48752, 'Normal_132': 52299, 'tumor_074': 12777,
                 'Normal_009': 5839, 'Tumor_034': 4172, 'Normal_107': 10436, 'Tumor_053': 9522, 'Tumor_068': 11244,
                 'Normal_106': 55690, 'Normal_031': 7160, 'Normal_030': 1839, 'Normal_027': 32, 'Tumor_001': 7755,
                 'Tumor_029': 8108, 'Normal_025': 8486, 'Normal_137': 19614, 'Tumor_041': 8221, 'Normal_058': 5778,
                 'Normal_046': 2204, 'Tumor_026': 18474, 'Normal_041': 7367, 'Tumor_042': 2818, 'Normal_032': 8294,
                 'Tumor_061': 13129, 'Normal_059': 1861, 'Normal_037': 7050, 'Tumor_019': 2214, 'tumor_099': 44288,
                 'Normal_159': 50970, 'Tumor_009': 11664, 'tumor_095': 27839, 'Normal_141': 17863, 'Normal_143': 53592,
                 'tumor_109': 51190, 'Normal_080': 15898, 'Tumor_056': 7862, 'Tumor_062': 7230, 'tumor_110': 43114,
                 'Normal_038': 578, 'Normal_033': 7632, 'Tumor_023': 4202, 'Normal_153': 32694, 'Tumor_066': 5454,
                 'Normal_102': 36353, 'Tumor_035': 3148, 'Normal_121': 14203, 'Normal_083': 3079, 'Tumor_047': 14378,
                 'Normal_022': 5534, 'Normal_155': 24138, 'Normal_003': 5443, 'Normal_110': 33385, 'Normal_017': 460,
                 'Normal_045': 154, 'Normal_006': 694, 'tumor_078': 58033, 'Tumor_004': 11716, 'Normal_123': 38147,
                 'Normal_011': 10229, 'Tumor_021': 16224, 'Normal_105': 10548, 'Tumor_027': 12576, 'Normal_075': 8683,
                 'tumor_090': 17492, 'tumor_108': 86857, 'Normal_093': 5112, 'Normal_084': 1661, 'Normal_005': 2297,
                 'Tumor_024': 13441, 'Normal_004': 1209, 'Normal_060': 1024, 'Normal_029': 1428, 'tumor_085': 40589}

heatmap_feature_names = ['region_count', 'ratio_tumor_tissue', 'largest_tumor_area', 'longest_axis_largest_tumor',
                         'pixels_gt_90', 'avg_prediction', 'max_area', 'mean_area', 'area_variance', 'area_skew',
                         'area_kurt', 'max_perimeter', 'mean_perimeter', 'perimeter_variance', 'perimeter_skew',
                         'perimeter_kurt', 'max_eccentricity', 'mean_eccentricity', 'eccentricity_variance',
                         'eccentricity_skew', 'eccentricity_kurt', 'max_extent', 'mean_extent', 'extent_variance',
                         'extent_skew', 'extent_kurt', 'max_solidity', 'mean_solidity', 'solidity_variance',
                         'solidity_skew', 'solidity_kurt', 'label']

BATCH_SIZE = 32

N_TRAIN_SAMPLES = 288000
N_VALIDATION_SAMPLES = 10000
N_SAMPLES_PER_TRAIN_SHARD = 1000
N_SAMPLES_PER_VALIDATION_SHARD = 250

NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX = 100
NUM_POSITIVE_PATCHES_FROM_EACH_BBOX = 500
PATCH_INDEX_NEGATIVE = 700000
PATCH_INDEX_POSITIVE = 700000

TUMOR_PROB_THRESHOLD = 0.90
PIXEL_WHITE = 255
PIXEL_BLACK = 0

HEATMAP_FEATURE_CSV_TRAIN = 'heatmap_features_train.csv'
HEATMAP_FEATURE_CSV_VALIDATION = 'heatmap_features_validation.csv'


def get_heatmap_ckpt_path(model_name):
    if user == 'arjun':
        return HEATMAP_MODEL_CKPT_PATH_ARJUN[model_name]
    else:
        return HEATMAP_MODEL_CKPT_PATH_MILLPC[model_name]


def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    return filename


def format_2f(number):
    return float("{0:.2f}".format(number))


def step_range(start, end, step):
    while start <= end:
        yield start
        start += step
