import glob
import os

import cv2
import numpy as np
from skimage.measure import label
from skimage.measure import regionprops

from camelyon16 import utils as utils

FILTER_DIM = 3


def extract_features(heatmap_prob, t=0.90):
    heatmap_prob[heatmap_prob < int(t * 255)] = 0
    heatmap_prob[heatmap_prob >= int(t * 255)] = 255
    close_kernel = np.ones((FILTER_DIM, FILTER_DIM), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(heatmap_prob), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((FILTER_DIM, FILTER_DIM), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    heatmap_prob = image_open[:, :, :1]
    heatmap_prob = np.reshape(heatmap_prob, (heatmap_prob.shape[0], heatmap_prob.shape[1]))

    labeled_img = label(heatmap_prob)
    region_props = regionprops(labeled_img)
    n_regions = len(region_props)
    print('No of regions: %d' % n_regions)
    for index in range(n_regions):
        # print('\n\nDisplaying region: %d' % index)
        region = region_props[index]
        # print('area: ', region['area'])
        # print('bbox: ', region['bbox'])
        # print('centroid: ', region['centroid'])
        # print('convex_area: ', region['convex_area'])
        # print('eccentricity: ', region['eccentricity'])
        # print('extent: ', region['extent'])
        # print('major_axis_length: ', region['major_axis_length'])
        # print('minor_axis_length: ', region['minor_axis_length'])
        # print('orientation: ', region['orientation'])
        # print('perimeter: ', region['perimeter'])
        # print('solidity: ', region['solidity'])

        cv2.rectangle(image_open, (region['bbox'][1], region['bbox'][0]),
                      (region['bbox'][3], region['bbox'][2]), color=(0, 255, 0),
                      thickness=1)
        cv2.ellipse(image_open, (int(region['centroid'][1]), int(region['centroid'][0])),
                    (int(region['major_axis_length'] / 2), int(region['minor_axis_length'] / 2)),
                    region['orientation'] * 90, 0, 360, color=(0, 0, 255),
                    thickness=2)

    cv2.imshow('bbox', image_open)
    key = cv2.waitKey(0) & 0xFF

    if key == 27:  # escape
        exit(0)


def extract_features_all(heatmap_prob_name_postfix):
    heatmap_prob_paths = glob.glob(os.path.join(utils.HEAT_MAP_DIR, '*umor*%s' % heatmap_prob_name_postfix))
    heatmap_prob_paths.sort()

    for heatmap_prob_path in heatmap_prob_paths:
        print('extracting features for: %s' % utils.get_filename_from_path(heatmap_prob_path))
        heatmap_prob = cv2.imread(heatmap_prob_path)
        cv2.imshow('heatmap_prob', heatmap_prob)
        extract_features(heatmap_prob)


# old_prob = cv2.imread('tumor_076_prob_old.png')
# extract_features(np.array(old_prob))
# # new_prob = cv2.imread('tumor_076_prob_new.png')
# # old_new_prob = np.array(old_prob)
# old_threshold = np.array(old_prob)
#
# # for row in range(old_prob.shape[0]):
# #     for col in range(old_prob.shape[1]):
# #         if old_prob[row, col, 0] >= 0.70*255 and new_prob[row, col, 0] < 0.50*255:
# #             old_new_prob[row, col, :] = new_prob[row, col, :]
#
#
# # old_new_prob[new_prob < 0.20*255] = 0
#
# old_threshold[old_threshold < int(0.90 * 255)] = 0
# old_threshold[old_threshold >= int(0.90 * 255)] = 255
# # new_prob[new_prob >= 0.51*255] = 255
# # old_prob[old_prob < 0.90*255] = 0
# # new_prob[new_prob < 0.51*255] = 0
#
# # cv2.imshow('old_prob', old_prob)
# # cv2.imshow('old_new_prob', old_new_prob)
#
#
# close_kernel = np.ones((FILTER_DIM, FILTER_DIM), dtype=np.uint8)
# image_close = cv2.morphologyEx(np.array(old_threshold), cv2.MORPH_CLOSE, close_kernel)
# open_kernel = np.ones((FILTER_DIM, FILTER_DIM), dtype=np.uint8)
# image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
# print(image_open.shape)
#
# # cv2.imshow('old_threshold', old_threshold)
# # cv2.imshow('image_close', image_close)
# cv2.imshow('image_open', image_open)
# # cv2.imshow('new_prob', new_prob)
# cv2.waitKey(0) & 0xFF


def extract_features_first_heatmap():
    extract_features_all(heatmap_prob_name_postfix='_prob.png')


def extract_features_second_heatmap():
    extract_features_all(heatmap_prob_name_postfix='_prob_%s.png' % utils.SECOND_HEATMAP_MODEL)


if __name__ == '__main__':
    extract_features_first_heatmap()
