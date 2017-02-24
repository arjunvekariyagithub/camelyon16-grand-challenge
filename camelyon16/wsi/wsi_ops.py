import cv2
import numpy as np
from PIL import Image
from openslide import OpenSlide, OpenSlideUnsupportedFormatError

import camelyon16.data as data


class WSIOps(object):
    """
        # ================================
        # Class to annotate WSIs with ROIs
        # ================================

    """
    negative_patch_index = 118456
    positive_patch_index = 2230
    def_level = 7

    def extract_patches_mask(self, wsi_image, wsi_mask, level_used, bounding_boxes):
        """

            Extract positive patches targeting annotated tumor region

            Save extracted patches to desk as .png image files

            :param wsi_image:
            :param wsi_mask:
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to tumor regions
            :return:
        """

        mag_factor = pow(2, level_used)

        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for i, bounding_box in enumerate(bounding_boxes):
            b_x_start = int(bounding_box[0]) * mag_factor
            b_y_start = int(bounding_box[1]) * mag_factor
            b_x_end = (int(bounding_box[0]) + int(bounding_box[2])) * mag_factor
            b_y_end = (int(bounding_box[1]) + int(bounding_box[3])) * mag_factor
            X = np.random.random_integers(b_x_start, high=b_x_end, size=500)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=500)
            # X = np.arange(b_x_start, b_x_end-256, 5)
            # Y = np.arange(b_y_start, b_y_end-256, 5)

            for x, y in zip(X, Y):
                mask = wsi_mask.read_region((x, y), 0, (data.PATCH_SIZE, data.PATCH_SIZE))
                mask_gt = np.array(mask)
                mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)

                white_pixel_cnt_gt = cv2.countNonZero(mask_gt)

                if white_pixel_cnt_gt > ((data.PATCH_SIZE * data.PATCH_SIZE) * 0.90):
                    patch = wsi_image.read_region((x, y), 0, (data.PATCH_SIZE, data.PATCH_SIZE))
                    patch.save(data.PROCESSED_PATCHES_USE_MASK_POSITIVE_PATH + data.PATCH_TUMOR_PREFIX +
                               str(self.positive_patch_index), 'PNG')
                    self.positive_patch_index += 1
                    patch.close()

                mask.close()

    def extract_patches_normal(self, wsi_image, level_used, bounding_boxes):
        """
            Extract negative patches from Normal WSIs

            Save extracted patches to desk as .png image files

            :param wsi_image:
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to detected ROIs
            :return:

        """

        mag_factor = pow(2, level_used)

        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for i, bounding_box in enumerate(bounding_boxes):
            b_x_start = int(bounding_box[0]) * mag_factor
            b_y_start = int(bounding_box[1]) * mag_factor
            b_x_end = (int(bounding_box[0]) + int(bounding_box[2])) * mag_factor
            b_y_end = (int(bounding_box[1]) + int(bounding_box[3])) * mag_factor
            X = np.random.random_integers(b_x_start, high=b_x_end, size=500)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=500)
            # X = np.arange(b_x_start, b_x_end-256, 5)
            # Y = np.arange(b_y_start, b_y_end-256, 5)

            for x, y in zip(X, Y):
                patch = wsi_image.read_region((x, y), 0, (data.PATCH_SIZE, data.PATCH_SIZE))
                patch_array = np.array(patch)

                patch_hsv = cv2.cvtColor(patch_array, cv2.COLOR_BGR2HSV)
                # [20, 20, 20]
                lower_red = np.array([20, 20, 20])
                # [255, 255, 255]
                upper_red = np.array([200, 200, 200])
                mask = cv2.inRange(patch_hsv, lower_red, upper_red)
                white_pixel_cnt = cv2.countNonZero(mask)

                if white_pixel_cnt > ((data.PATCH_SIZE * data.PATCH_SIZE) * 0.50):
                    patch.save(data.PROCESSED_PATCHES_NORMAL_NEGATIVE_PATH + data.PATCH_NORMAL_PREFIX +
                               str(self.negative_patch_index), 'PNG')
                    self.negative_patch_index += 1

                patch.close()

    def extract_patches_tumor(self, wsi_image, wsi_mask, level_used, bounding_boxes):
        """
            From Tumor WSIs, extract both, negative patches from Normal area and positive patches from Tumor area

            Save extracted patches to desk as .png image files

            :param wsi_image:
            :param wsi_mask:
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to tumor regions
            :return:
            
        """

        mag_factor = pow(2, level_used)

        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for i, bounding_box in enumerate(bounding_boxes):
            b_x_start = int(bounding_box[0]) * mag_factor
            b_y_start = int(bounding_box[1]) * mag_factor
            b_x_end = (int(bounding_box[0]) + int(bounding_box[2])) * mag_factor
            b_y_end = (int(bounding_box[1]) + int(bounding_box[3])) * mag_factor
            X = np.random.random_integers(b_x_start, high=b_x_end, size=500)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=500)
            # X = np.arange(b_x_start, b_x_end-256, 5)
            # Y = np.arange(b_y_start, b_y_end-256, 5)

            for x, y in zip(X, Y):
                patch = wsi_image.read_region((x, y), 0, (data.PATCH_SIZE, data.PATCH_SIZE))
                mask = wsi_mask.read_region((x, y), 0, (data.PATCH_SIZE, data.PATCH_SIZE))
                mask_gt = np.array(mask)
                # mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)
                mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)
                patch_array = np.array(patch)

                white_pixel_cnt_gt = cv2.countNonZero(mask_gt)

                if white_pixel_cnt_gt == 0:  # mask_gt does not contain tumor area
                    patch_hsv = cv2.cvtColor(patch_array, cv2.COLOR_BGR2HSV)
                    lower_red = np.array([20, 20, 20])
                    upper_red = np.array([200, 200, 200])
                    mask_patch = cv2.inRange(patch_hsv, lower_red, upper_red)
                    white_pixel_cnt = cv2.countNonZero(mask_patch)

                    if white_pixel_cnt > ((data.PATCH_SIZE * data.PATCH_SIZE) * 0.50):
                        patch.save(data.PROCESSED_PATCHES_TUMOR_NEGATIVE_PATH + data.PATCH_NORMAL_PREFIX +
                                   str(self.negative_patch_index), 'PNG')
                        self.negative_patch_index += 1
                else:  # mask_gt contains tumor area
                    if white_pixel_cnt_gt >= ((data.PATCH_SIZE * data.PATCH_SIZE) * 0.85):
                        patch.save(data.PROCESSED_PATCHES_TUMOR_POSITIVE_PATH + data.PATCH_TUMOR_PREFIX +
                                   str(self.positive_patch_index), 'PNG')
                        self.positive_patch_index += 1

                patch.close()
                mask.close()

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

    def read_wsi_normal(self, wsi_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            wsi_image = OpenSlide(wsi_path)
            level_used = min(self.def_level, wsi_image.level_count - 1)

            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None

        return wsi_image, rgb_image

    def read_wsi_tumor(self, wsi_path, mask_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            wsi_image = OpenSlide(wsi_path)
            wsi_mask = OpenSlide(mask_path)

            level_used = min(self.def_level, wsi_image.level_count - 1, wsi_mask.level_count - 1)

            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))
            wsi_mask.close()
        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None

        return wsi_image, rgb_image, level_used

    def find_roi_bb_mask(self, mask_image):

        mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        bounding_boxes = self.get_bb_mask(np.array(mask))
        return bounding_boxes

    def find_roi_bb_normal(self, rgb_image):

        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        # [20, 20, 20]
        lower_red = np.array([20, 50, 20])
        # [255, 255, 255]
        upper_red = np.array([200, 150, 200])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        close_kernel = np.ones((25, 25), dtype=np.uint8)
        image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
        open_kernel = np.ones((30, 30), dtype=np.uint8)
        image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel))
        bounding_boxes = self.get_bb_normal(np.array(image_open))
        return bounding_boxes

    def find_roi_bb_tumor(self, rgb_image):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        close_kernel = np.ones((50, 50), dtype=np.uint8)
        image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
        open_kernel = np.ones((30, 30), dtype=np.uint8)
        image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel))
        bounding_boxes = self.get_bb_tumor(np.array(image_open))
        return bounding_boxes

    @staticmethod
    def get_bb_mask(cont_img):
        _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        return bounding_boxes

    @staticmethod
    def get_bb_normal(cont_img):
        _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        return bounding_boxes

    @staticmethod
    def get_bb_tumor(cont_img):
        _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        return bounding_boxes
