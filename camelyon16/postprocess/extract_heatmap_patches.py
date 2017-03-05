from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.insert(0, '/home/arjun/MS/Thesis/CAMELYON-16/source')
import glob
import os.path
import threading

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from camelyon16 import utils as utils
from camelyon16.ops.wsi_ops import WSIOps


class Queue(object):
    def __init__(self):
        self.patches = []
        self.pos = []

    def put(self, patch, pos):
        self.patches.append(patch)
        self.pos.append(pos)
        assert self.patches.__len__() == self.pos.__len__(), 'Both queue lengths must be same.'

    def get_next_batch(self):
        assert self.patches.__len__() == self.pos.__len__(), 'Both queue lengths must be same.'
        cnt = utils.BATCH_SIZE if self.patches.__len__() > utils.BATCH_SIZE else self.patches.__len__()
        items = self.patches[:cnt]
        del self.patches[:cnt]
        del self.pos[:cnt]
        return items

    def clear(self):
        self.patches = []
        self.pos = []

    def print(self):
        print(self.pos)


def extract_patch_from_bb(thread_index, bounding_box, wsi_image, level_used, heat_map_dir):
    # factor to map low res cords into high res
    mag_factor = pow(2, level_used)
    b_x_start = int(bounding_box[0])
    b_y_start = int(bounding_box[1])
    b_x_end = (int(bounding_box[0]) + int(bounding_box[2]))
    b_y_end = (int(bounding_box[1]) + int(bounding_box[3]))
    x_cords = np.arange(b_x_start, b_x_end)
    y_cords = np.arange(b_y_start, b_y_end)
    print('Apx. patch count for thread(%d): %d' % (thread_index, len(x_cords) * len(y_cords)))
    # print(min(x_cords), max(x_cords))
    # print(min(y_cords), max(y_cords))

    for y in y_cords:
        for x in x_cords:
            x_large = x * mag_factor
            y_large = y * mag_factor
            wsi_patch = wsi_image.read_region((x_large, y_large), 0, (utils.PATCH_SIZE, utils.PATCH_SIZE))
            file_name = str(x) + '_' + str(y) + '_' + str(level_used)
            wsi_patch.save(os.path.join(heat_map_dir, file_name), 'PNG')

            # patch = np.array(wsi_patch)
            # # print('Thread(%d) - processing: (%d, %d)' % (thread_index, x, y))
            # patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            # lower_red = np.array([20, 20, 20])
            # upper_red = np.array([200, 200, 200])
            # mask = cv2.inRange(patch_hsv, lower_red, upper_red)
            # white_pixel_cnt = cv2.countNonZero(mask)
            #
            # if white_pixel_cnt > ((utils.PATCH_SIZE * utils.PATCH_SIZE) * 0.50):
            #     # print('Thread(%d) ************ accepted **************' % thread_index)
            #     file_name = str(x) + '_' + str(y) + '_' + str(level_used) + '_accept'
            #     Image.fromarray(mask).save(os.path.join(heat_map_dir, file_name), 'PNG')
            # else:
            #     file_name = str(x) + '_' + str(y) + '_' + str(level_used) + '_rej'
            #     Image.fromarray(mask).save(os.path.join(heat_map_dir, file_name), 'PNG')

            wsi_patch.close()


def extract_patches(wsi_image_path, wsi_mask_path, wsi_image_name):
    print('extract_patches(): %s' % wsi_image_name)

    '''
      tmp test: 068,


    '''

    wsi_image, rgb_image, tumor_gt_mask, level_used = wsi_ops.read_wsi_tumor(wsi_image_path, wsi_mask_path)

    assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % wsi_image_name

    heatmap_patch_dir = os.path.join(utils.HEAT_MAP_RAW_PATCHES_DIR, wsi_image_name)
    # print(heatmap__patch_dir)
    if not os.path.exists(heatmap_patch_dir):
        os.makedirs(heatmap_patch_dir)

    bounding_boxes, contour, mask_contour, bbox, image_open = \
        wsi_ops.find_roi_bb_tumor(np.array(rgb_image), tumor_gt_mask)

    # Image.fromarray(rgb_image).save(os.path.join(utils.HEAT_MAP_WSIs_PATH, wsi_image_name), 'PNG')
    # Image.fromarray(contour).save(os.path.join(utils.HEAT_MAP_WSIs_PATH, wsi_image_name + '_contour'), 'PNG')
    # Image.fromarray(bbox).save(os.path.join(utils.HEAT_MAP_WSIs_PATH, wsi_image_name + '_bbox'), 'PNG')
    # Image.fromarray(mask_contour).save(os.path.join(utils.HEAT_MAP_WSIs_PATH, wsi_image_name + '_mask'), 'PNG')
    print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))
    # for bbox in bounding_boxes[:1]:
    #     extract_patch_from_bb(bbox, wsi_image, level_used)

    return
    coord = tf.train.Coordinator()

    threads = []
    for thread_index in range(len(bounding_boxes)):
        args = (thread_index, bounding_boxes[thread_index], wsi_image, level_used, heatmap_patch_dir)
        t = threading.Thread(target=extract_patch_from_bb, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    wsi_image.close()
    sys.stdout.flush()


if __name__ == '__main__':
    # dataset = Dataset(DATA_SET_NAME, data_subset[1])
    # evaluate(dataset)
    wsi_ops = WSIOps()
    wsi_image_names = glob.glob(os.path.join(utils.TUMOR_WSI_PATH, '*.tif'))
    wsi_image_names.sort()
    wsi_mask_names = glob.glob(os.path.join(utils.TUMOR_MASK_PATH, '*.tif'))
    wsi_mask_names.sort()

    image_mask_pair = zip(wsi_image_names, wsi_mask_names)
    image_mask_pair = list(image_mask_pair)
    # image_mask_pair = image_mask_pair[67:68]
    for image_path, mask_path in image_mask_pair:
        extract_patches(image_path, mask_path, utils.get_filename_from_path(image_path))
