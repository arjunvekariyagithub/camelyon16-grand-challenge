import csv
import glob
import os
import random

import cv2
import numpy as np
import scipy.stats.stats as st
from skimage.measure import label
from skimage.measure import regionprops

import matplotlib.pyplot as plt
from camelyon16 import utils as utils
from camelyon16.ops.wsi_ops import WSIOps


def prob_to_heatmap(prob, heatmap_path):
    prob = np.array(prob[:, :, :1], np.float32)
    prob_2d = np.reshape(prob, (prob.shape[0], prob.shape[1]))
    prob_2d /= 255
    h_flip_prob = cv2.flip(prob_2d, 0)
    plt.imshow(h_flip_prob, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.clim(0.00, 1.00)
    plt.axis([0, h_flip_prob.shape[1], 0, h_flip_prob.shape[0]])
    plt.savefig(heatmap_path)
    plt.clf()


if __name__ == '__main__':
    heatmap_prob_paths = glob.glob(
        os.path.join(utils.HEAT_MAP_DIR, '*%s*' % 'Test*prob.png'))
    heatmap_prob_paths.sort()

    for heatmap_prob_path in heatmap_prob_paths:
        print('processing: %s' % utils.get_filename_from_path(heatmap_prob_path))
        heatmap_prob = cv2.imread(heatmap_prob_path)
        heatmap_path = heatmap_prob_path.replace('prob', 'heatmap')
        print(heatmap_prob_path)
        print(heatmap_path)
        prob_to_heatmap(heatmap_prob, heatmap_path)
