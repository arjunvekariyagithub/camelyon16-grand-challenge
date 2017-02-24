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
"""Read and preprocess image data.

 Image processing occurs on a single image at a time. Image are read and
 preprocessed in parallel across multiple threads. The resulting images
 are concatenated together to form a single batch for training or evaluation.

 -- Provide processed image data for a network:
 inputs: Construct batches of evaluation examples of images.
 distorted_inputs: Construct batches of training examples of images.
 batch_inputs: Construct batches of training or evaluation examples of images.

 -- Data processing:
 parse_example_proto: Parses an Example proto containing a training example
   of an image.

 -- Image decoding:
 decode_jpeg: Decode a JPEG encoded string into a 3-D float32 Tensor.

 -- Image preprocessing:
 image_preprocessing: Decode and preprocess one image for evaluation or training
 distort_image: Distort one image for training a network.
 eval_image: Prepare one image for evaluation.
 distort_color: Distort the color in one image for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from camelyon16.inception.dataset import Dataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
# Arjun - updated
tf.app.flags.DEFINE_integer('image_size', 256,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")

# Images are preprocessed asynchronously using multiple threads specified by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing on a given Inception tower. A larger shuffling queue guarantees
# better mixing across examples within a batch and results in slightly higher
# predictive performance in a trained model. Empirically,
# --input_queue_memory_factor=16 works well. A value of 16 implies a queue size
# of 1024*16 images. Assuming RGB 299x299 images, this implies a queue size of
# 16GB. If the machine is memory limited, then decrease this factor to
# decrease the CPU memory footprint, accordingly.
# Arjun - updated (16 -> 4)
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 4,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")


def decode_png(image_buffer, scope=None):
    """Decode a PNG string into one 3-D float image Tensor.

    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for op_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(scope, 'decode_png', [image_buffer]):
        # Decode the string as an RGB PNG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_png. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_png(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def image_preprocessing(image_buffer):
    """Decode and preprocess one image for evaluation or training.

    Args:
      image_buffer: PNG encoded string Tensor
      train: boolean
      thread_id: integer indicating preprocessing thread

    Returns:
      3-D float Tensor containing an appropriately scaled image

    Raises:
      ValueError: if user does not provide bounding box
    """

    image = decode_png(image_buffer)

    # height = FLAGS.image_size
    # width = FLAGS.image_size
    image = tf.reshape(image, shape=[FLAGS.image_size, FLAGS.image_size, 3])

    tf.image.per_image_standardization(image)

    # subtract channel wise mean
    mean = tf.reduce_mean(image, axis=[0, 1])
    mean = tf.reshape(mean, [1, 1, 3])
    image = tf.subtract(image, mean)

    # Arjun - updated
    # if train:
    #     image = distort_image(image, height, width, thread_id)
    # else:
    #     image = eval_image(image, height, width)

    # Arjun - updated -> already in range [0, 1] so no need to do it again
    # First, scale scale to [0, 1) and finally scale to [-1,1]
    # image = tf.divide(image, 255)
    # image = tf.subtract(image, 0.5)
    # image = tf.multiply(image, 2.0)
    return image, mean


def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_tf_records.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:

      image/height: 462
      image/width: 581
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 615
      image/class/synset: 'n03623198'
      image/class/text: 'knee pad'
      image/object/bbox/xmin: 0.1
      image/object/bbox/xmax: 0.9
      image/object/bbox/ymin: 0.2
      image/object/bbox/ymax: 0.6
      image/object/bbox/label: 615
      image/format: 'JPEG'
      image/filename: 'ILSVRC2012_val_00041207.JPEG'
      image/encoded: <JPEG encoded string>

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.

    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      text: Tensor tf.string containing the human-readable label.
    """
    # Dense features in Example proto.
    # Arjun - updated
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                                default_value=0)
    }

    features = tf.parse_single_example(example_serialized, feature_map)
    print(features)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    return features['image/encoded'], label


def main(unused_argv):
    dataset = Dataset('Camelyon', 'train')

    with tf.Session() as sess:
        data_files = dataset.data_files_test()
        print(data_files)
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        filename_queue = tf.train.string_input_producer(data_files,
                                                        shuffle=False,
                                                        capacity=1)

        reader = dataset.reader()
        _, example_serialized = reader.read(filename_queue)

        images_and_labels = []
        for thread_id in range(1):
            # Parse a serialized Example proto to extract the image and metadata.
            image_buffer, label_index = parse_example_proto(
                example_serialized)
            im, mean = image_preprocessing(image_buffer)
            image = tf.cast(im, tf.float32)
            images_and_labels.append([image, label_index])

        images, label_index_batch = tf.train.batch_join(
            images_and_labels,
            batch_size=3,
            capacity=2 * 1 * 3)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        images = sess.run(images)
        print(images.shape)
        print(images)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
