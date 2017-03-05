# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts image data to TFRecords file format with Example protos.

The image data set is expected to reside in JPEG files located in the
following directory structure.

  data_dir/label_0/image0.jpeg
  data_dir/label_0/image1.jpg
  ...
  data_dir/label_1/weird-image.jpeg
  data_dir/label_1/my-image.jpeg
  ...

where the sub-directory is the unique label associated with these images.

This tf script converts the training and evaluation data into
a sharded data set consisting of TFRecord files

  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-00127-of-01024

and

  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128

where we have selected 1024 and 128 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:

  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always'JPEG'

  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [0, num_labels] where 0 is unused and left as
    the background class.
  image/class/text: string specifying the human-readable version of the label
    e.g. 'dog'

If you data set involves bounding boxes, please look at build_imagenet_data.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '/home/arjun/MS/Thesis/CAMELYON-16/source')
import os
import time
from datetime import datetime
import math

import tensorflow as tf

from camelyon16 import utils as utils

N_TRAIN_SAMPLES = 250000
N_VALIDATION_SAMPLES = 10000
N_SAMPLES_PER_TRAIN_SHARD = 1000
N_SAMPLES_PER_VALIDATION_SHARD = 250

tf.app.flags.DEFINE_string('output_directory', utils.HEAT_MAP_TF_RECORDS_DIR,
                           'Output data directory')

tf.app.flags.DEFINE_integer('num_shards', 1,
                            'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 5,
                            'Number of threads to preprocess the images.')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer, patch_name):
    """
        Build an Example proto for an example.

    """

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/patch_name': _bytes_feature(tf.compat.as_bytes(patch_name)),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example


class ImageCoder(object):
    """Helper class that provides tf image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def decode_png(self, image_data):
        image = self._sess.run(self._decode_png,
                               feed_dict={self._decode_png_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _is_png(filename):
    """Determine if a file contains a PNG format image.

    Args:
      filename: string, path of the image file.

    Returns:
      boolean indicating if the image is a PNG.
    """
    return '.png' in filename


def _process_image(patch_path, coder):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide tf image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(patch_path, 'r') as f:
        image_data = f.read()

    # Decode the RGB PNG.
    image = coder.decode_png(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _process_patches(name, patch_paths, patch_names, wsi_filename):
    """Process and save list of images as TFRecord of Example protos.

    Args:
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    assert len(patch_paths) == len(patch_names)

    sys.stdout.flush()

    # Create a generic tf-based utility for converting all image codings.
    coder = ImageCoder()

    output_filename = '%s-patches-%s' % (name, wsi_filename)
    output_dir = os.path.join(FLAGS.output_directory, wsi_filename)
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    counter = 0
    start_time = time.time()
    for patch_path, patch_name in zip(patch_paths, patch_names):
        image_buffer, height, width = _process_image(patch_path, coder)

        example = _convert_to_example(image_buffer, patch_name)
        writer.write(example.SerializeToString())
        counter += 1

        if not counter % 1000:
            duration = time.time() - start_time
            print('%d secs: Processed %d of %d images.' %
                  (math.ceil(duration), counter, len(patch_paths)))
            sys.stdout.flush()
            start_time = time.time()

    writer.close()

    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(patch_paths)))
    sys.stdout.flush()


def _find_patches(data_dir):
    """Build a list of all images files and labels in the data set.

    Args:
      data_dir: string, path to the root directory of images.

    """
    print('Determining list of file paths and names from %s.' % data_dir)

    file_names = []
    file_paths = []

    # Construct the list of PNG file paths and names.
    png_file_path = '%s/*' % data_dir
    matching_files = tf.gfile.Glob(png_file_path)

    file_names.extend(os.listdir(data_dir))
    file_paths.extend(matching_files)

    file_paths = sorted(file_paths)
    file_names = sorted(file_names)

    assert len(file_paths) == len(file_names)

    print('Found %d PNG files' % len(file_names))
    return file_paths, file_names


def _process_dataset(name, directory, wsi_filename):
    """Process a complete data set and save it as a TFRecord.

    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
    """
    patch_paths, patch_names = _find_patches(directory)
    _process_patches(name, patch_paths, patch_names, wsi_filename)


def main(unused_argv):
    # assert not FLAGS.num_shards % FLAGS.num_threads, (
    #     'Please make the FLAGS.num_threads commensurate with FLAGS.num_shards')
    print('Saving results to %s' % FLAGS.output_directory)

    raw_patches_file_names = sorted(os.listdir(utils.HEAT_MAP_RAW_PATCHES_DIR))
    print(raw_patches_file_names)
    raw_patches_file_names = raw_patches_file_names[67:68]

    for wsi_filename in raw_patches_file_names:
        print('processing: %s' % wsi_filename)
        raw_patches_dir = os.path.join(utils.HEAT_MAP_RAW_PATCHES_DIR, wsi_filename)
        assert os.path.exists(raw_patches_dir), 'directory %s does not exist' % raw_patches_dir
        _process_dataset('heatmap', raw_patches_dir, wsi_filename)


if __name__ == '__main__':
    tf.app.run()
