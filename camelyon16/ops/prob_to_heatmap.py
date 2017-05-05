import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from camelyon16 import utils as utils


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
    heatmap_prob_paths_first_model = glob.glob(
        os.path.join(utils.HEAT_MAP_DIR, '*%s*' % '*umor*prob.png'))
    heatmap_prob_paths_first_model.sort()

    for heatmap_prob_path_first_model in heatmap_prob_paths_first_model:
        wsi_name = utils.get_filename_from_path(heatmap_prob_path_first_model)
        wsi_name_tokens = wsi_name.split('_')
        wsi_name = wsi_name_tokens[0] + '_' + wsi_name_tokens[1]
        print('processing: %s' % wsi_name)
        heatmap_prob_path_second_model = glob.glob(
            os.path.join(utils.HEAT_MAP_DIR, '*%s*%s' % (wsi_name, '_prob_%s.png' % utils.SECOND_HEATMAP_MODEL)))
        heatmap_prob_second_model = cv2.imread(heatmap_prob_path_second_model[0])
        heatmap_prob_first_model = cv2.imread(heatmap_prob_path_first_model)

        for row in range(heatmap_prob_first_model.shape[0]):
            for col in range(heatmap_prob_first_model.shape[1]):
                if heatmap_prob_first_model[row, col, 0] >= 0.70 * 255 and \
                                heatmap_prob_second_model[row, col, 0] < 0.25 * 255:
                    heatmap_prob_first_model[row, col, :] = heatmap_prob_second_model[row, col, :]

        heatmap_path = heatmap_prob_path_first_model.replace('prob', 'heatmap_ensemble')
        # print(heatmap_prob_path_first_model)
        # print(heatmap_prob_path_second_model)
        # print(heatmap_path)
        prob_to_heatmap(heatmap_prob_first_model, heatmap_path)
