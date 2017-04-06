import cv2
import numpy as np
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from PIL import Image

import camelyon16.utils as utils


class PatchExtractor(object):
    @staticmethod
    def extract_positive_patches_from_tumor_region(wsi_image, tumor_gt_mask, level_used,
                                                   bounding_boxes, patch_save_dir, patch_prefix,
                                                   patch_index):
        """

            Extract positive patches targeting annotated tumor region

            Save extracted patches to desk as .png image files

            :param wsi_image:
            :param tumor_gt_mask:
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to tumor regions
            :param patch_save_dir: directory to save patches into
            :param patch_prefix: prefix for patch name
            :param patch_index:
            :return:
        """

        mag_factor = pow(2, level_used)
        tumor_gt_mask = cv2.cvtColor(tumor_gt_mask, cv2.COLOR_BGR2GRAY)
        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for bounding_box in bounding_boxes:
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1])
            b_x_end = int(bounding_box[0]) + int(bounding_box[2])
            b_y_end = int(bounding_box[1]) + int(bounding_box[3])
            X = np.random.random_integers(b_x_start, high=b_x_end, size=utils.NUM_POSITIVE_PATCHES_FROM_EACH_BBOX)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=utils.NUM_POSITIVE_PATCHES_FROM_EACH_BBOX)

            for x, y in zip(X, Y):
                if int(tumor_gt_mask[y, x]) != 0:
                    x_large = x * mag_factor
                    y_large = y * mag_factor
                    patch = wsi_image.read_region((x_large, y_large), 0, (utils.PATCH_SIZE, utils.PATCH_SIZE))
                    patch.save(patch_save_dir + patch_prefix + str(patch_index), 'PNG')
                    patch_index += 1
                    patch.close()

        return patch_index

    @staticmethod
    def extract_negative_patches_from_normal_wsi(wsi_image, image_open, level_used,
                                                 bounding_boxes, patch_save_dir, patch_prefix,
                                                 patch_index):
        """
            Extract negative patches from Normal WSIs

            Save extracted patches to desk as .png image files

            :param wsi_image:
            :param image_open:
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to detected ROIs
            :param patch_save_dir: directory to save patches into
            :param patch_prefix: prefix for patch name
            :param patch_index:
            :return:

        """

        mag_factor = pow(2, level_used)

        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for bounding_box in bounding_boxes:
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1])
            b_x_end = int(bounding_box[0]) + int(bounding_box[2])
            b_y_end = int(bounding_box[1]) + int(bounding_box[3])
            X = np.random.random_integers(b_x_start, high=b_x_end, size=utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)

            for x, y in zip(X, Y):
                if int(image_open[y, x]) == 1:
                    x_large = x * mag_factor
                    y_large = y * mag_factor
                    patch = wsi_image.read_region((x_large, y_large), 0, (utils.PATCH_SIZE, utils.PATCH_SIZE))
                    patch.save(patch_save_dir + patch_prefix + str(patch_index), 'PNG')
                    patch_index += 1
                    patch.close()

        return patch_index

    @staticmethod
    def extract_negative_patches_from_tumor_wsi(wsi_image, tumor_gt_mask, image_open, level_used,
                                                bounding_boxes, patch_save_dir, patch_prefix,
                                                patch_index):
        """
            From Tumor WSIs extract negative patches from Normal area (reject tumor area)
            Save extracted patches to desk as .png image files

            :param wsi_image:
            :param tumor_gt_mask:
            :param image_open: morphological open image of wsi_image
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to tumor regions
            :param patch_save_dir: directory to save patches into
            :param patch_prefix: prefix for patch name
            :param patch_index:
            :return:

        """

        mag_factor = pow(2, level_used)
        tumor_gt_mask = cv2.cvtColor(tumor_gt_mask, cv2.COLOR_BGR2GRAY)
        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for bounding_box in bounding_boxes:
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1])
            b_x_end = int(bounding_box[0]) + int(bounding_box[2])
            b_y_end = int(bounding_box[1]) + int(bounding_box[3])
            X = np.random.random_integers(b_x_start, high=b_x_end, size=utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)

            for x, y in zip(X, Y):
                if int(image_open[y, x]) == 1:
                    x_large = x * mag_factor
                    y_large = y * mag_factor
                    if int(tumor_gt_mask[y, x]) == 0:  # mask_gt does not contain tumor area
                        patch = wsi_image.read_region((x_large, y_large), 0, (utils.PATCH_SIZE, utils.PATCH_SIZE))
                        patch.save(patch_save_dir + patch_prefix + str(patch_index), 'PNG')
                        patch_index += 1
                        patch.close()

        return patch_index


class WSIOps(object):
    """
        # ================================
        # Class to annotate WSIs with ROIs
        # ================================

    """

    def_level = 7

    @staticmethod
    def read_wsi_mask(mask_path, level=def_level):
        try:
            wsi_mask = OpenSlide(mask_path)

            mask_image = np.array(wsi_mask.read_region((0, 0), level,
                                                       wsi_mask.level_dimensions[level]))

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None

        return wsi_mask, mask_image

    @staticmethod
    def read_wsi_normal(wsi_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            wsi_image = OpenSlide(wsi_path)
            level_used = wsi_image.level_count - 1
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None, None

        return wsi_image, rgb_image, level_used

    @staticmethod
    def read_wsi_tumor(wsi_path, mask_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            wsi_image = OpenSlide(wsi_path)
            wsi_mask = OpenSlide(mask_path)

            level_used = wsi_image.level_count - 1

            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))

            mask_level = wsi_mask.level_count - 1
            mask_image = wsi_mask.read_region((0, 0), mask_level,
                                              wsi_image.level_dimensions[mask_level])
            resize_factor = float(1.0 / pow(2, level_used - mask_level))
            # print('resize_factor: %f' % resize_factor)
            mask_image = cv2.resize(np.array(mask_image), (0, 0), fx=resize_factor, fy=resize_factor)

            wsi_mask.close()
        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None, None, None

        return wsi_image, rgb_image, mask_image, level_used

    def find_roi_bbox_tumor_gt_mask(self, mask_image):
        mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        bounding_boxes = self.get_bbox_mask(np.array(mask))
        return bounding_boxes

    def find_roi_bbox_normal(self, rgb_image):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([200, 200, 200])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        close_kernel = np.ones((20, 20), dtype=np.uint8)
        image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
        open_kernel = np.ones((5, 5), dtype=np.uint8)
        image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
        bounding_boxes, rgb_contour = self.get_bbox_normal(image_open, rgb_image)
        return bounding_boxes, image_open

    def find_roi_bbox_tumor(self, rgb_image, tumor_gt_mask):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([200, 200, 200])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        close_kernel = np.ones((20, 20), dtype=np.uint8)
        image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
        open_kernel = np.ones((5, 5), dtype=np.uint8)
        image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
        bounding_boxes, rgb_contour, mask_contour = self.get_bbox_tumor(image_open, rgb_image, tumor_gt_mask)

        # rgb_bbox = self.draw_bbox(rgb_image, bounding_boxes)
        # rgb_bbox_split = self.split_bbox(rgb_image, bounding_boxes, image_open)


        # Image.fromarray(rgb_image).save(os.path.join(utils.HEAT_MAP_WSIs_PATH, wsi_image_name), 'PNG')
        # Image.fromarray(rgb_contour).save(os.path.join(utils.HEAT_MAP_WSIs_PATH, wsi_image_name + '_contour'), 'PNG')
        # Image.fromarray(rgb_bbox).save(os.path.join(utils.HEAT_MAP_WSIs_PATH, wsi_image_name + '_bbox'), 'PNG')
        # Image.fromarray(mask_contour).save(os.path.join(utils.HEAT_MAP_WSIs_PATH, wsi_image_name + '_mask'), 'PNG')

        cv2.imshow('contour', rgb_contour)
        # cv2.imshow('contour_mask', mask_contour)
        # cv2.imshow('bbox', rgb_bbox)
        # cv2.imshow('image_open', rgb_bbox_split)
        # cv2.imshow('mask', mask)
        cv2.waitKey(0) & 0xFF

        return bounding_boxes, image_open

    @staticmethod
    def get_bbox_mask(cont_img):
        _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        return bounding_boxes

    @staticmethod
    def get_bbox_normal(cont_img, image):
        rgb_contour = image.copy()
        _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(rgb_contour, contours, -1, line_color, 2)

        bounding_boxes = [cv2.boundingRect(c) for c in contours]

        return bounding_boxes, rgb_contour

    @staticmethod
    def get_bbox_tumor(cont_img, image, tumor_gt_mask):
        rgb_contour = image.copy()
        mask_contour = tumor_gt_mask.copy()
        _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(rgb_contour, contours, -1, line_color, 2)
        cv2.drawContours(mask_contour, contours, -1, line_color, 2)

        bounding_boxes = [cv2.boundingRect(c) for c in contours]

        return bounding_boxes, rgb_contour, mask_contour

    @staticmethod
    def draw_bbox(image, bounding_boxes):
        rgb_bbox = image.copy()
        for i, bounding_box in enumerate(bounding_boxes):
            x = int(bounding_box[0])
            y = int(bounding_box[1])
            cv2.rectangle(rgb_bbox, (x, y), (x + bounding_box[2], y + bounding_box[3]), color=(0, 0, 255),
                          thickness=2)
        return rgb_bbox

    @staticmethod
    def split_bbox(image, bounding_boxes, image_open):
        rgb_bbox_split = image.copy()
        for bounding_box in bounding_boxes:
            for x in range(bounding_box[0], bounding_box[0] + bounding_box[2]):
                for y in range(bounding_box[1], bounding_box[1] + bounding_box[3]):
                    if int(image_open[y, x]) == 1:
                        cv2.rectangle(rgb_bbox_split, (x, y), (x, y),
                                      color=(255, 0, 0), thickness=2)

        return rgb_bbox_split
