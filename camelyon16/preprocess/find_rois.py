import glob
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import camelyon16.utils as utils


class WSI(object):
    """
        # ================================
        # Class to annotate WSIs with ROIs
        # ================================

    """
    index = 0
    wsi_paths = []
    mask_paths = []
    def_level = 7
    key = 0

    def read_normal_wsi(self, wsi_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            self.wsi_path = wsi_path
            self.wsi_image = OpenSlide(wsi_path)

            level = min(self.def_level, self.wsi_image.level_count - 1)
            print('level used: %d' % level)
            print(self.wsi_image.level_dimensions[level])

            self.rgb_image_pil = self.wsi_image.read_region((0, 0), level,
                                                            self.wsi_image.level_dimensions[level])
            self.rgb_image = np.array(self.rgb_image_pil)

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return False

        return True

    def read_tumor_wsi(self, wsi_path, mask_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            self.wsi_path = wsi_path
            self.wsi_image = OpenSlide(wsi_path)
            self.mask_image = OpenSlide(mask_path)

            level_used = self.wsi_image.level_count - 1

            self.rgb_image_pil = self.wsi_image.read_region((0, 0), level_used,
                                                            self.wsi_image.level_dimensions[level_used])
            self.rgb_image = np.array(self.rgb_image_pil)

            mask_level = self.mask_image.level_count - 1
            self.rgb_mask_pil = self.mask_image.read_region((0, 0), mask_level,
                                                            self.mask_image.level_dimensions[mask_level])

            resize_factor = float(1.0 / pow(2, level_used - mask_level))
            self.mask = cv2.resize(np.array(self.rgb_mask_pil), (0, 0), fx=resize_factor, fy=resize_factor)
            # self.rgb_image = cv2.resize(self.rgb_image, (0, 0), fx=0.50, fy=0.50)
        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return False

        return True

    def find_roi_normal(self):
        # self.mask = cv2.cvtColor(self.mask, cv2.CV_32SC1)
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        # [20, 20, 20]
        lower_red = np.array([30, 30, 30])
        # [255, 255, 255]
        upper_red = np.array([200, 200, 200])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(self.rgb_image, self.rgb_image, mask=mask)

        # (50, 50)
        close_kernel = np.ones((50, 50), dtype=np.uint8)
        close_kernel_tmp = np.ones((30, 30), dtype=np.uint8)
        image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
        image_close_tmp = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel_tmp))
        # (30, 30)
        open_kernel = np.ones((30, 30), dtype=np.uint8)
        open_kernel_tmp = np.ones((30, 30), dtype=np.uint8)
        image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel))
        image_open_tmp = Image.fromarray(cv2.morphologyEx(np.array(image_close_tmp), cv2.MORPH_OPEN, open_kernel_tmp))
        contour_rgb, bounding_boxes, contour_rgb_tmp = self.get_normal_image_contours(np.array(image_open),
                                                                                      self.rgb_image,
                                                                                      np.array(image_open_tmp))
        # self.draw_bbox(bounding_boxes)

        self.display(contour_rgb, contour_rgb_tmp)

    def find_roi_tumor(self):
        # self.mask = cv2.cvtColor(self.mask, cv2.CV_32SC1)
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([200, 200, 200])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(self.rgb_image, self.rgb_image, mask=mask)

        # (50, 50)
        close_kernel = np.ones((20, 20), dtype=np.uint8)
        close_kernel_tmp = np.ones((50, 50), dtype=np.uint8)
        image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
        image_close_tmp = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel_tmp))
        # (30, 30)
        open_kernel = np.ones((5, 5), dtype=np.uint8)
        open_kernel_tmp = np.ones((30, 30), dtype=np.uint8)
        image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel))
        image_open_tmp = Image.fromarray(cv2.morphologyEx(np.array(image_close_tmp), cv2.MORPH_OPEN, open_kernel_tmp))
        contour_rgb, contour_mask, bounding_boxes, contour_rgb_tmp, contour_mask_tmp = self.get_tumor_image_contours(
            np.array(image_open), self.rgb_image,
            self.mask, np.array(image_open_tmp))

        wsi_name = utils.get_filename_from_path(self.wsi_path)
        # cv2.imwrite(os.path.join(utils.THESIS_FIGURE_DIR, wsi_name) + '_hsv_mask.png', mask)
        # cv2.imwrite(os.path.join(utils.THESIS_FIGURE_DIR, wsi_name) + '_mask.png', self.mask)
        # cv2.imwrite(os.path.join(utils.THESIS_FIGURE_DIR, wsi_name) + '_image_close.png', np.array(image_close))
        # cv2.imwrite(os.path.join(utils.THESIS_FIGURE_DIR, wsi_name) + '_image_open.png', np.array(image_open))

        # rgb_bbox = self.rgb_image.copy()
        # for i, bounding_box in enumerate(bounding_boxes):
        #     x = int(bounding_box[0])
        #     y = int(bounding_box[1])
        #     cv2.rectangle(rgb_bbox, (x, y), (x + bounding_box[2], y + bounding_box[3]), color=(0, 0, 255),
        #                   thickness=2)

        # cv2.imshow('contour_rgb', contour_rgb)
        # cv2.imshow('contour_bbox', rgb_bbox)
        # cv2.imwrite(os.path.join(utils.THESIS_FIGURE_DIR, wsi_name) + 'contour.png', contour_rgb)
        # cv2.imwrite(os.path.join(utils.THESIS_FIGURE_DIR, wsi_name) + '_bbox.png', rgb_bbox)
        cv2.imwrite(os.path.join(utils.THESIS_FIGURE_DIR, wsi_name) + '_hsv.png', hsv)
        cv2.imshow('hsv', hsv)
        cv2.waitKey(0)

        # self.display(contour_rgb, contour_rgb_tmp, contour_mask)

    def get_normal_image_contours(self, cont_img, rgb_image, cont_img_tmp):
        _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours_tmp, _ = cv2.findContours(cont_img_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        # print(boundingBoxes)
        contours_rgb_image_array = np.array(rgb_image)
        contours_rgb_image_array_tmp = np.array(rgb_image)

        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(contours_rgb_image_array, contours, -1, line_color, 3)
        cv2.drawContours(contours_rgb_image_array_tmp, contours_tmp, -1, line_color, 3)
        # cv2.drawContours(mask_image, contours_mask, -1, line_color, 3)
        return contours_rgb_image_array, boundingBoxes, contours_rgb_image_array_tmp

    def get_tumor_image_contours(self, cont_img, rgb_image, mask_image, cont_img_tmp):
        _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours_tmp, _ = cv2.findContours(cont_img_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # _, contours_mask, _ = cv2.findContours(mask_image, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        # print(boundingBoxes)
        contours_rgb_image_array = np.array(rgb_image)
        contours_rgb_image_array_tmp = np.array(rgb_image)
        contours_mask_image_array = np.array(mask_image)
        contours_mask_image_array_tmp = np.array(mask_image)

        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(contours_rgb_image_array, contours, -1, line_color, 3)
        cv2.drawContours(contours_rgb_image_array_tmp, contours_tmp, -1, line_color, 3)
        cv2.drawContours(contours_mask_image_array, contours, -1, line_color, 3)
        cv2.drawContours(contours_mask_image_array_tmp, contours_tmp, -1, line_color, 3)
        # cv2.drawContours(mask_image, contours_mask, -1, line_color, 3)
        return contours_rgb_image_array, contours_mask_image_array, boundingBoxes, contours_rgb_image_array_tmp, \
               contours_mask_image_array_tmp

    def display(self, contour_rgb, contour_rgb_tmp, contour_mask=None):
        # cv2.imshow('rgb', self.rgb_image)
        # cv2.imshow('mask', mask)
        # cv2.imshow('image_close',np.array(image_close))
        # cv2.imshow('image_open', np.array(image_open))

        contour_rgb = cv2.resize(contour_rgb, (0, 0), fx=0.60, fy=0.60)
        cv2.imshow('contour_rgb', np.array(contour_rgb))
        contour_rgb_tmp = cv2.resize(contour_rgb_tmp, (0, 0), fx=0.60, fy=0.60)
        cv2.imshow('contour_rgb_tmp', np.array(contour_rgb_tmp))
        if contour_mask is not None:
            contour_mask = cv2.resize(contour_mask, (0, 0), fx=0.60, fy=0.60)
            cv2.imshow('contour_mask', np.array(contour_mask))

        # contour_mask_tmp = cv2.resize(contour_mask_tmp, (0, 0), fx=0.40, fy=0.40)
        # cv2.imshow('contour_mask_tmp', np.array(contour_mask_tmp))
        # cv2.imshow('contour_mask_only', np.array(contour_mask_only))
        # cv2.imshow('wsi_mask', self.mask)
        # cv2.imshow('self_bb', np.array(self.rgb_image_pil))

    def draw_bbox(self, bounding_boxes):
        draw = ImageDraw.Draw(self.rgb_image_pil)
        for i, bounding_box in enumerate(bounding_boxes):
            x = int(bounding_box[0])
            y = int(bounding_box[1])
            thickness = 5
            for offset in range(thickness):
                draw.rectangle([x + offset, y + offset, x + offset + bounding_box[2], y + offset + bounding_box[3]],
                               outline=(255, 0, 0))

    def wait(self):
        self.key = cv2.waitKey(0) & 0xFF
        print('key: %d' % self.key)

        if self.key == 27:  # escape
            return False
        elif self.key == 81:  # <- (prev)
            self.index -= 1
            if self.index < 0:
                self.index = len(self.wsi_paths) - 1
        elif self.key == 83:  # -> (next)
            self.index += 1
            if self.index >= len(self.wsi_paths):
                self.index = 0

        return True


def run_on_tumor_data():
    wsi = WSI()
    wsi.wsi_paths = glob.glob(os.path.join(utils.TUMOR_WSI_PATH, '*.tif'))
    wsi.wsi_paths.sort()
    wsi.mask_paths = glob.glob(os.path.join(utils.TUMOR_MASK_PATH, '*.tif'))
    wsi.mask_paths.sort()
    # improved -
    # interesting cases - 43, (71, 73, 84) - diff levels,
    # self better -
    # self worst -
    # check - 15, 31, 34, 44, 47, 54, 57, 64, 75, 76,

    wsi.index = 109
    wsi_path = wsi.wsi_paths[wsi.index]
    mask_path = wsi.mask_paths[wsi.index]
    if wsi.read_tumor_wsi(wsi_path, mask_path):
        wsi.find_roi_tumor()

    # while True:
    #     wsi_path = wsi.wsi_paths[wsi.index]
    #     mask_path = wsi.mask_paths[wsi.index]
    #     print(wsi_path)
    #     print(mask_path)
    #     if wsi.read_tumor_wsi(wsi_path, mask_path):
    #         wsi.find_roi_tumor()
    #         if not wsi.wait():
    #             break
    #     else:
    #         if wsi.key == 81:
    #             wsi.index -= 1
    #             if wsi.index < 0:
    #                 wsi.index = len(wsi.wsi_paths) - 1
    #         elif wsi.key == 83:
    #             wsi.index += 1
    #             if wsi.index >= len(wsi.wsi_paths):
    #                 wsi.index = 0


def run_on_normal_data():
    wsi = WSI()
    wsi.wsi_paths = glob.glob(os.path.join(utils.NORMAL_WSI_PATH, '*.tif'))
    wsi.wsi_paths.sort()
    # improved -
    # interesting cases - 43, (71, 73, 84) - diff levels,
    # self better -
    # self worst -
    # check - 15, 31, 34, 44, 47, 54, 57, 64, 75, 76,

    wsi.index = 0
    # wsi_paths = ops.wsi_paths[30:]
    # ops.read_wsi(WSI_FILE_NAME)
    # ops.find_roi()

    while True:
        wsi_path = wsi.wsi_paths[wsi.index]
        print(wsi_path)
        if wsi.read_normal_wsi(wsi_path):
            wsi.find_roi_normal()
            if not wsi.wait():
                break
        else:
            if wsi.key == 81:
                wsi.index -= 1
                if wsi.index < 0:
                    wsi.index = len(wsi.wsi_paths) - 1
            elif wsi.key == 83:
                wsi.index += 1
                if wsi.index >= len(wsi.wsi_paths):
                    wsi.index = 0


if __name__ == '__main__':
    run_on_tumor_data()
    # run_on_normal_data()
