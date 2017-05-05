# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 18:25:03 2017
@author: Francesco Ciompi
"""
import os

import numpy as np
from PIL import Image
import glob

"""
    Stain normalization of test images of colorectal cancer tissue samples
    used in [1]. The images were first used in [2], and can be downloaded
    from this link:

    https://zenodo.org/record/53169/files/Kather_texture_2016_larger_images_10.zip

    Change the path in 'crc'images_folder' to point to the folder with the
    images downloded.

    Run the script to apply the stain normalization algorithm described in [3]
    to the ten test images.

    Applying stain normalization consists in converting the RGB values of the
    input image using a look-up table. The values in the LUT have been learned
    by applying the method in [3] to each input image separately, and using
    a whole slide image as a template. A low-resolution version of the
    image used as template is provided in this folder as 'rc_template.png'

    ---------------------------------------------------------------------------
    References

    [1] F.Ciompi et al. "The importance of stain normalization in colorectal
    tissue classification with convolutional networks". ISBI 2017

    [2] JN Kather et al. "Multi-class texture analysis in colorectal cancer
    histology", Scientific Reports, 2016

    [3] B. Ehteshami Bejnordi et al., "Stain specific standardization of
    whole-slide histopathological images". IEEE Transaction on Medical Imaging,
    35(2), 404-415, 2016.

"""


def apply_lut(tile, lut):
    """ Apply look-up-table to tile to normalize H&E staining. """
    ps = tile.shape  # tile size is (rows, cols, channels)
    reshaped_tile = tile.reshape((ps[0] * ps[1], 3))
    normalized_tile = np.zeros((ps[0] * ps[1], 3))
    idxs = range(ps[0] * ps[1])
    Index = 256 * 256 * reshaped_tile[idxs, 0] + 256 * reshaped_tile[idxs, 1] + reshaped_tile[idxs, 2]
    normalized_tile[idxs] = lut[Index.astype(int)]
    return normalized_tile.reshape(ps[0], ps[1], 3).astype(np.uint8)


if __name__ == "__main__":

    crc_images_folder = '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/stain_normalization/Kather_texture_2016_larger_images_10'

    file_paths = glob.glob(os.path.join(crc_images_folder, 'CRC*APPLICATION.tif'))
    file_paths.sort()
    # filenames = ['CRC-Prim-HE-01_APPLICATION.tif',
    #              'CRC-Prim-HE-02_APPLICATION.tif',
    #              'CRC-Prim-HE-03_APPLICATION.tif',
    #              'CRC-Prim-HE-04_APPLICATION.tif',
    #              'CRC-Prim-HE-05_APPLICATION.tif',
    #              'CRC-Prim-HE-06_APPLICATION.tif',
    #              'CRC-Prim-HE-07_APPLICATION.tif',
    #              'CRC-Prim-HE-08_APPLICATION.tif',
    #              'CRC-Prim-HE-09_APPLICATION.tif',
    #              'CRC-Prim-HE-10_APPLICATION.tif']

    for file_path in file_paths:
        tile = np.asarray(Image.open(file_path))
        lut_file_path = os.path.splitext(file_path)[0] + '_LUT.tif'
        lut = np.asarray(Image.open(lut_file_path)).squeeze()
        print('applying stain normalization to file {}...'.format(file_path))
        tile_normalized = apply_lut(tile, lut)

        im = Image.fromarray(tile_normalized.astype(np.uint8))
        im.save(os.path.splitext(file_path)[0] + '_normalized.tif')
print('image saved to disk')
