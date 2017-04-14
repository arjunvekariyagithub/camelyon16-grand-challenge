import getpass

users = ['arjun', 'millpc']

user = getpass.getuser()
print('user: %s' % user)

assert user in users, 'User not Authorised!!'


def is_running_on_server():
    return user == 'arjun'


data_subset = ['train', 'train-aug', 'validation', 'validation-aug', 'heatmap']

THESIS_FIGURE_DIR = '/home/millpc/Documents/Arjun/Study/Thesis/Defense/Me/figures/'

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

tpr_harvard = [0.72, 0.72, 0.74, 0.74, 0.78, 0.78, 0.84, 0.84, 0.86, 0.86, 0.91, 0.91, 0.925, 0.925, 0.935, 0.935,
               0.955, 1.0]
fpr_harvard = [0.0, 0.03, 0.03, 0.05, 0.05, 0.09, 0.09, 0.12, 0.12, 0.135, 0.135, 0.16, 0.16, 0.24, 0.24, 0.78, 0.78,
               1.0]

tpr_exb = [0.48, 0.48, 0.55, 0.55, 0.62, 0.62, 0.78, 0.78, 0.8, 0.8, 0.82, 0.82, 0.84, 0.84, 0.87, 0.87, 0.9, 0.9, 0.92,
           0.92, 0.95, 0.95, 0.98, 1.0]
fpr_exb = [0.0, 0.02, 0.02, 0.042, 0.042, 0.05, 0.05, 0.085, 0.085, 0.09, 0.09, 0.115, 0.115, 0.17, 0.17, 0.22, 0.22,
           0.25, 0.25, 0.48, 0.48, 0.58, 1.0, 1.0]

tpr_quincy_wong = [0.03, 0.63, 0.63, 0.655, 0.655, 0.72, 0.72, 0.77, 0.77, 0.78, 0.78, 0.8, 0.8, 0.825, 0.825,
                   0.84, 0.84, 0.85, 0.85, 0.92, 0.92, 0.95, 0.95, 0.965, 1.0]
fpr_quincy_wong = [0.0, 0.03, 0.04, 0.04, 0.07, 0.07, 0.165, 0.165, 0.22, 0.22, 0.24, 0.24, 0.25, 0.25, 0.26,
                   0.26, 0.44, 0.44, 0.48, 0.48, 0.55, 0.55, 0.6, 1.0, 1.0]

tpr_mtu = [0.48, 0.48, 0.56, 0.56, 0.6, 0.6, 0.64, 0.64, 0.67, 0.67, 0.72, 0.72, 0.75, 0.75, 0.78, 0.78, 0.8, 0.8,
           0.825, 0.825, 0.84, 0.84, 0.86, 0.86, 0.9, 0.9, 0.92, 0.92, 0.95, 0.95, 0.965, 0.965, 0.98, 1.0]
fpr_mtu = [0.0, 0.02, 0.02, 0.045, 0.045, 0.06, 0.06, 0.08, 0.08, 0.115, 0.115, 0.13, 0.13, 0.135, 0.135, 0.23, 0.23,
           0.25, 0.25, 0.26, 0.26, 0.4, 0.4, 0.475, 0.475, 0.495, 0.495, 0.66, 0.66, 0.765, 0.765, 0.8, 1.0, 1.0]

tpr_nlp = [0.525, 0.525, 0.62, 0.62, 0.64, 0.64, 0.66, 0.66, 0.72, 0.72, 0.74, 0.74, 0.76, 0.76, 0.78, 0.78, 0.8, 0.8,
           0.82, 0.82, 0.84, 0.84, 0.88, 0.88, 0.915, 0.915, 0.94, 0.94, 0.965, 0.965, 0.98, 0.98, 1.0, 1.0]
fpr_nlp = [0.0, 0.02, 0.02, 0.06, 0.06, 0.16, 0.16, 0.22, 0.22, 0.29, 0.29, 0.31, 0.31, 0.36, 0.36, 0.372, 0.372, 0.42,
           0.42, 0.44, 0.44, 0.48, 0.48, 0.58, 0.58, 0.72, 0.72, 0.735, 0.735, 0.8, 0.8, 0.9, 1.0, 1.0]


test_wsi_names = ['Test_001', 'Test_002', 'Test_003', 'Test_004', 'Test_005', 'Test_006', 'Test_007', 'Test_008',
                  'Test_009', 'Test_010', 'Test_011', 'Test_012', 'Test_013', 'Test_014', 'Test_015', 'Test_016',
                  'Test_017', 'Test_018', 'Test_019', 'Test_020', 'Test_021', 'Test_022', 'Test_023', 'Test_024',
                  'Test_025', 'Test_026', 'Test_027', 'Test_028', 'Test_029', 'Test_030', 'Test_031', 'Test_032',
                  'Test_033', 'Test_034', 'Test_035', 'Test_036', 'Test_037', 'Test_038', 'Test_039', 'Test_040',
                  'Test_041', 'Test_042', 'Test_043', 'Test_044', 'Test_045', 'Test_046', 'Test_047', 'Test_048',
                  'Test_049', 'Test_050', 'Test_051', 'Test_052', 'Test_053', 'Test_054', 'Test_055', 'Test_056',
                  'Test_057', 'Test_058', 'Test_059', 'Test_060', 'Test_061', 'Test_062', 'Test_063', 'Test_064',
                  'Test_065', 'Test_066', 'Test_067', 'Test_068', 'Test_069', 'Test_070', 'Test_071', 'Test_072',
                  'Test_073', 'Test_074', 'Test_075', 'Test_076', 'Test_077', 'Test_078', 'Test_079', 'Test_080',
                  'Test_081', 'Test_082', 'Test_083', 'Test_084', 'Test_085', 'Test_086', 'Test_087', 'Test_088',
                  'Test_089', 'Test_090', 'Test_091', 'Test_092', 'Test_093', 'Test_094', 'Test_095', 'Test_096',
                  'Test_097', 'Test_098', 'Test_099', 'Test_100', 'Test_101', 'Test_102', 'Test_103', 'Test_104',
                  'Test_105', 'Test_106', 'Test_107', 'Test_108', 'Test_109', 'Test_110', 'Test_111', 'Test_112',
                  'Test_113', 'Test_114', 'Test_115', 'Test_116', 'Test_117', 'Test_118', 'Test_119', 'Test_120',
                  'Test_121', 'Test_122', 'Test_123', 'Test_124', 'Test_125', 'Test_126', 'Test_127', 'Test_128',
                  'Test_129', 'Test_130']

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
                 'Tumor_024': 13441, 'Normal_004': 1209, 'Normal_060': 1024, 'Normal_029': 1428, 'tumor_085': 40589,
                 'Test_006': 11158, 'Test_032': 12246, 'Test_060': 3181, 'Test_028': 13624, 'Test_100': 20669,
                 'Test_099': 58864, 'Test_051': 5983, 'Test_075': 9918, 'Test_084': 22604, 'Test_053': 17792,
                 'Test_062': 5029, 'Test_096': 2252, 'Test_031': 59070, 'Test_008': 61463, 'Test_118': 3790,
                 'Test_111': 8485, 'Test_114': 65086, 'Test_045': 4711, 'Test_107': 1690, 'Test_022': 60004,
                 'Test_106': 4608, 'Test_064': 1965, 'Test_016': 82343, 'Test_039': 1504, 'Test_083': 2364,
                 'Test_078': 60505, 'Test_035': 8859, 'Test_015': 3052, 'Test_130': 5635, 'Test_113': 81479,
                 'Test_034': 10027, 'Test_013': 3431, 'Test_050': 11604, 'Test_094': 49443, 'Test_063': 59411,
                 'Test_108': 6529, 'Test_066': 20188, 'Test_011': 11803, 'Test_072': 4451, 'Test_121': 4510,
                 'Test_021': 16930, 'Test_126': 10102, 'Test_068': 15660, 'Test_042': 35203, 'Test_097': 14196,
                 'Test_105': 107890, 'Test_029': 2859, 'Test_101': 16480, 'Test_074': 11568, 'Test_019': 30449,
                 'Test_033': 39451, 'Test_012': 53978, 'Test_069': 44135, 'Test_030': 6901, 'Test_040': 4914,
                 'Test_089': 10790, 'Test_079': 36463, 'Test_043': 3623, 'Test_054': 9608, 'Test_065': 12714,
                 'Test_010': 16887, 'Test_091': 2683, 'Test_041': 59003, 'Test_109': 13054, 'Test_061': 8249,
                 'Test_071': 19054, 'Test_052': 6537, 'Test_104': 13175, 'Test_092': 87630, 'Test_001': 48559,
                 'Test_119': 2236, 'Test_005': 3565, 'Test_103': 5953, 'Test_059': 6892, 'Test_020': 1612,
                 'Test_122': 31691, 'Test_124': 2362, 'Test_038': 6958, 'Test_125': 51569, 'Test_073': 10029,
                 'Test_098': 51339, 'Test_004': 45887, 'Test_037': 102060, 'Test_110': 55723, 'Test_025': 1862,
                 'Test_117': 8356, 'Test_056': 18028, 'Test_088': 4452, 'Test_082': 12560, 'Test_076': 13676,
                 'Test_027': 5818, 'Test_023': 8119, 'Test_002': 8579, 'Test_095': 33768, 'Test_102': 9717,
                 'Test_049': 14853, 'Test_044': 30987, 'Test_086': 44188, 'Test_026': 77775, 'Test_093': 4203,
                 'Test_024': 5653, 'Test_057': 3703, 'Test_129': 2764, 'Test_123': 3508, 'Test_120': 17075,
                 'Test_112': 3862, 'Test_018': 23417, 'Test_009': 6655, 'Test_070': 3913, 'Test_046': 16594,
                 'Test_047': 33913, 'Test_003': 5076, 'Test_077': 62140, 'Test_058': 72536, 'Test_116': 36564,
                 'Test_081': 5931, 'Test_085': 2633, 'Test_127': 27610, 'Test_036': 6250, 'Test_055': 21480,
                 'Test_115': 1878, 'Test_090': 21989, 'Test_048': 72674, 'Test_080': 64701, 'Test_007': 4694,
                 'Test_087': 5326, 'Test_128': 17838, 'Test_067': 13710, 'Test_014': 83605, 'Test_017': 6787}

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

HEATMAP_FEATURE_CSV_TRAIN = 'features/heatmap_features_train.csv'
HEATMAP_FEATURE_CSV_VALIDATION = 'features/heatmap_features_validation.csv'
HEATMAP_FEATURE_CSV_TRAIN_ALL = 'features/heatmap_features_train_all.csv'
HEATMAP_FEATURE_CSV_TEST = 'features/heatmap_features_test.csv'
HEATMAP_FEATURE_CSV_TRAIN_SECOND_MODEL = 'features/heatmap_features_train_model8.csv'
HEATMAP_FEATURE_CSV_VALIDATION_SECOND_MODEL = 'features/heatmap_features_validation_model8.csv'
HEATMAP_FEATURE_CSV_TRAIN_ALL_SECOND_MODEL = 'features/heatmap_features_train_all_model8.csv'
HEATMAP_FEATURE_CSV_TEST_SECOND_MODEL = 'features/heatmap_features_test_model8.csv'

TEST_CSV_GT = 'GT.csv'


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
