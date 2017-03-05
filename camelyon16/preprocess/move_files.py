import os
from shutil import copyfile

CAMELYON_PROCESSED_PATCHES_TRAIN = '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/Processed/' \
                                   'patch-based-classification/raw-data/train/'
CAMELYON_PROCESSED_PATCHES_TRAIN_NEGATIVE = CAMELYON_PROCESSED_PATCHES_TRAIN + 'label-0/'
CAMELYON_PROCESSED_PATCHES_TRAIN_POSITIVE = CAMELYON_PROCESSED_PATCHES_TRAIN + 'label-1/'

CAMELYON_PROCESSED_PATCHES_VALIDATION = '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/' \
                                        'Processed/patch-based-classification/raw-data/validation/'
CAMELYON_PROCESSED_PATCHES_VALIDATION_NEGATIVE = CAMELYON_PROCESSED_PATCHES_VALIDATION + 'label-0/'
CAMELYON_PROCESSED_PATCHES_VALIDATION_POSITIVE = CAMELYON_PROCESSED_PATCHES_VALIDATION + 'label-1/'

N_POSITIVE_SAMPLES_TO_MOVE = 2214
N_NEGATIVE_SAMPLES_TO_MOVE = 5000


def copy_files(from_dir, to_dir, n_sample):
    print('copy_files()')
    file_names = os.listdir(to_dir)
    file_names = sorted(file_names)
    file_names = file_names[:n_sample]
    for file_name in file_names:
        copyfile(to_dir + file_name, from_dir + file_name + '_aug')
        # os.rename(to_dir + file_name + '.png', from_dir + file_name + '_aug' + '.png')


def move_files(from_dir, to_dir, n_sample):
    print('move_positive_samples()')
    pos_file_names = os.listdir(to_dir)
    pos_file_names = sorted(pos_file_names, reverse=True)
    pos_file_names = pos_file_names[:n_sample]

    for file_name in pos_file_names:
        os.rename(to_dir + file_name, from_dir + file_name)


copy_files(CAMELYON_PROCESSED_PATCHES_TRAIN_POSITIVE, CAMELYON_PROCESSED_PATCHES_TRAIN_POSITIVE, 22209)
