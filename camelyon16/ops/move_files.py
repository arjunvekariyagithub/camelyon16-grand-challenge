import os
from shutil import copyfile
import camelyon16.utils as utils
import numpy as np

N_POSITIVE_SAMPLES_TO_MOVE = 2214
N_NEGATIVE_SAMPLES_TO_MOVE = 5000


def copy_files(from_dir, to_dir, n_sample=0):
    print('copy_files()')
    file_names = os.listdir(from_dir)
    file_names = sorted(file_names)
    assert len(file_names) > n_sample, "Not enough files to copy. available: %d, requested: %d" % \
                                       (len(file_names), n_sample)

    indices = np.random.random_integers(0, high=len(file_names)-1, size=n_sample)

    for index in indices:
        copyfile(from_dir + file_names[index], to_dir + file_names[index] + '_aug')


def move_files(from_dir, to_dir, n_sample):
    print('move_positive_samples()')
    file_names = os.listdir(from_dir)
    file_names = sorted(file_names)
    assert len(file_names) > n_sample, "Not enough files to move. available: %d, requested: %d" % \
                                       (len(file_names), n_sample)

    indices = np.random.random_integers(0, high=len(file_names) - 1, size=n_sample)

    for index in indices:
        os.rename(from_dir + file_names[index], to_dir + file_names[index])


copy_files(utils.PATCHES_VALIDATION_NEGATIVE_PATH, utils.PATCHES_VALIDATION_AUG_NEGATIVE_PATH, n_sample=5)
