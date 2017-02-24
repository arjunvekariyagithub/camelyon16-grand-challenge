
DATA_DIR = '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/'

TUMOR_WSI_PATH = DATA_DIR + 'TrainingData/Train_Tumor'
NORMAL_WSI_PATH = DATA_DIR + 'TrainingData/Train_Normal'
TUMOR_MASK_PATH = DATA_DIR + 'TrainingData/Ground_Truth/Mask'
PROCESSED_PATCHES_TRAIN_NEGATIVE_PATH = DATA_DIR + 'Processed/patch-based-classification/raw-data/train/label-0/'
PROCESSED_PATCHES_TRAIN_POSITIVE_PATH = DATA_DIR + 'Processed/patch-based-classification/raw-data/train/label-1/'
PROCESSED_PATCHES_VALIDATION_NEGATIVE_PATH = DATA_DIR + \
                                             'Processed/patch-based-classification/raw-data/validation/label-0/'
PROCESSED_PATCHES_VALIDATION_POSITIVE_PATH = DATA_DIR + \
                                             'Processed/patch-based-classification/raw-data/validation/label-1/'
TF_RECORDS_PATH = DATA_DIR + 'Processed/patch-based-classification/tf-records/'

PATCH_SIZE = 256
PATCH_NORMAL_PREFIX = 'normal_'
PATCH_TUMOR_PREFIX = 'tumor_'

# dummy
PROCESSED_PATCHES_NORMAL_NEGATIVE_PATH = '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/' \
                                         'Processed/patch-based-classification/normal-label-0/'
PROCESSED_PATCHES_TUMOR_NEGATIVE_PATH = '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/' \
                                         'Processed/patch-based-classification/tumor-label-0/'
PROCESSED_PATCHES_TUMOR_POSITIVE_PATH = '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/' \
                                        'Processed/patch-based-classification/label-1/'
PROCESSED_PATCHES_USE_MASK_POSITIVE_PATH = '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/' \
                                        'Processed/patch-based-classification/use-mask-label-1/'


