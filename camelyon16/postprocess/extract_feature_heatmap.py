import glob
import os

import cv2
import numpy as np
from skimage.measure import label
from skimage.measure import regionprops

from camelyon16 import utils as utils
from camelyon16.ops.wsi_ops import WSIOps

FILTER_DIM = 2


def get_region_props(heatmap_prob):
    heatmap_prob_2d = heatmap_prob[:, :, :1]
    heatmap_prob_2d = np.reshape(heatmap_prob_2d, (heatmap_prob_2d.shape[0], heatmap_prob_2d.shape[1]))
    labeled_img = label(heatmap_prob_2d)
    return regionprops(labeled_img)


def draw_bbox(heatmap_threshold, region_props, threshold_label='t90'):
    n_regions = len(region_props)
    print('No of regions(%s): %d' % (threshold_label, n_regions))
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

        cv2.rectangle(heatmap_threshold, (region['bbox'][1], region['bbox'][0]),
                      (region['bbox'][3], region['bbox'][2]), color=(0, 255, 0),
                      thickness=1)
        cv2.ellipse(heatmap_threshold, (int(region['centroid'][1]), int(region['centroid'][0])),
                    (int(region['major_axis_length'] / 2), int(region['minor_axis_length'] / 2)),
                    region['orientation'] * 90, 0, 360, color=(0, 0, 255),
                    thickness=2)

    cv2.imshow('bbox_%s' % threshold_label, heatmap_threshold)


def get_largest_tumor_index(region_props):
    largest_tumor_index = -1

    largest_tumor_area = -1

    n_regions = len(region_props)
    for index in range(n_regions):
        if region_props[index]['area'] > largest_tumor_area:
            largest_tumor_area = region_props[index]['area']
            largest_tumor_index = index

    return largest_tumor_index


def get_longest_axis_in_largest_tumor_region(region_props, largest_tumor_region_index):
    largest_tumor_region = region_props[largest_tumor_region_index]
    return max(largest_tumor_region['major_axis_length'], largest_tumor_region['minor_axis_length'])


def get_tumor_region_to_tissue_ratio(region_props, image_open):
    tissue_area = cv2.countNonZero(image_open)
    tumor_area = 0

    n_regions = len(region_props)
    for index in range(n_regions):
        tumor_area += region_props[index]['area']

    return float(tumor_area) / tissue_area


def get_tumor_region_to_bbox_ratio(region_props):
    print()


def extract_features(heatmap_prob, image_open):
    heatmap_threshold_t90 = np.array(heatmap_prob)
    heatmap_threshold_t50 = np.array(heatmap_prob)
    heatmap_threshold_t90[heatmap_threshold_t90 < int(0.90 * 255)] = 0
    heatmap_threshold_t90[heatmap_threshold_t90 >= int(0.90 * 255)] = 255
    heatmap_threshold_t50[heatmap_threshold_t50 <= int(0.50 * 255)] = 0
    heatmap_threshold_t50[heatmap_threshold_t50 > int(0.50 * 255)] = 255

    # cv2.imshow('heatmap_threshold', heatmap_prob)
    # close_kernel = np.ones((FILTER_DIM, FILTER_DIM), dtype=np.uint8)
    # image_close = cv2.morphologyEx(np.array(heatmap_prob), cv2.MORPH_CLOSE, close_kernel)
    # cv2.imshow('image_close', image_close)
    # open_kernel = np.ones((FILTER_DIM, FILTER_DIM), dtype=np.uint8)
    # image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    # cv2.imshow('image_open', image_open)

    region_props_t90 = get_region_props(heatmap_threshold_t90)
    region_props_t50 = get_region_props(heatmap_threshold_t50)

    draw_bbox(np.array(heatmap_threshold_t90), region_props_t90, threshold_label='t90')
    draw_bbox(np.array(heatmap_threshold_t50), region_props_t50, threshold_label='t50')

    largest_tumor_region_index_t90 = get_largest_tumor_index(region_props_t90)
    largest_tumor_region_index_t50 = get_largest_tumor_index(region_props_t50)

    # Feature 10: given t = 0.5, the longest axis in the largest tumor region
    largest_axis_largest_tumor_region_t90 = get_longest_axis_in_largest_tumor_region(region_props_t90,
                                                                                     largest_tumor_region_index_t90)
    largest_axis_largest_tumor_region_t50 = get_longest_axis_in_largest_tumor_region(region_props_t50,
                                                                                     largest_tumor_region_index_t50)

    # Feature 05: given t = 0.5, the area of largest tumor region
    larget_tumor_region_area_t50 = region_props_t50[largest_tumor_region_index_t50]['area']
    larget_tumor_region_area_t90 = region_props_t90[largest_tumor_region_index_t90]['area']

    # Feature 04: ratio of tumor region when t = 0.9 to the tissue region
    tumor_region_to_tissue_ratio_t90 = get_tumor_region_to_tissue_ratio(region_props_t90, image_open)

    # cv2.imshow('heatmap_threshold_t90', heatmap_threshold_t90)
    # cv2.imshow('heatmap_threshold_t50', heatmap_threshold_t50)

    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # escape
        exit(0)


def extract_features_all(heatmap_prob_name_postfix):
    tumor_wsi_paths = glob.glob(os.path.join(utils.TUMOR_WSI_PATH, '*.tif'))
    tumor_wsi_paths.sort()
    normal_wsi_paths = glob.glob(os.path.join(utils.NORMAL_WSI_PATH, '*.tif'))
    normal_wsi_paths.sort()

    wsi_paths = tumor_wsi_paths + normal_wsi_paths
    print(len(wsi_paths))

    for wsi_path in wsi_paths:
        wsi_name = utils.get_filename_from_path(wsi_path)
        print('extracting features for: %s' % wsi_name)
        heatmap_prob_path = glob.glob(
            os.path.join(utils.HEAT_MAP_DIR, '*%s*%s' % (wsi_name, heatmap_prob_name_postfix)))
        print(heatmap_prob_path)
        image_open = wsi_ops.get_image_open(wsi_path)
        heatmap_prob = cv2.imread(heatmap_prob_path[0])
        # cv2.imshow('heatmap_prob', heatmap_prob)
        extract_features(heatmap_prob, image_open)


# old_prob = cv2.imread('tumor_076_prob_old.png')
# extract_features(np.array(old_prob))
# # new_prob = cv2.imread('tumor_076_prob_new.png')
# # old_new_prob = np.array(old_prob)
# old_threshold = np.array(old_prob)
#
# # for row in range(old_prob.shape[0]):
# #     for col in range(old_prob.shape[1]):
# #         if old_prob[row, col, 0] >= 0.70*255 and new_prob[row, col, 0] < 0.t50*255:
# #             old_new_prob[row, col, :] = new_prob[row, col, :]
#
#
# # old_new_prob[new_prob < 0.20*255] = 0
#
# old_threshold[old_threshold < int(0.t90 * 255)] = 0
# old_threshold[old_threshold >= int(0.t90 * 255)] = 255
# # new_prob[new_prob >= 0.51*255] = 255
# # old_prob[old_prob < 0.t90*255] = 0
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
    wsi_ops = WSIOps()
    extract_features_first_heatmap()
