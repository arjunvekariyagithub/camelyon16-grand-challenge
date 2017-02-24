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
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import math
import os.path
import time
from datetime import datetime

import numpy as np
import sklearn as sk
import tensorflow as tf
import cv2

from camelyon16 import data as data
from camelyon16.wsi.wsi_ops import WSIOps

FLAGS = tf.app.flags.FLAGS

TRAIN_DIR = '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/training/model3'
EVAL_DIR = '/home/arjun/MS/Thesis/CAMELYON-16/Data/Processed/evaluation'
CKPT_PATH = None

DATA_SET_NAME = 'TF-Records'
data_subset = ['train', 'validation']

tf.app.flags.DEFINE_string('eval_dir', EVAL_DIR,
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', TRAIN_DIR,
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.
                            We have 10000 examples.""")
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")

# tf.app.flags.DEFINE_integer('batch_size', 40,
#                             """Number of images to process in a batch.""")

BATCH_SIZE = 40


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
        cnt = BATCH_SIZE if self.patches.__len__() > BATCH_SIZE else self.patches.__len__()
        items = self.patches[:cnt]
        del self.patches[:cnt]
        del self.pos[:cnt]
        return items

    def clear(self):
        self.patches = []
        self.pos = []

    def print(self):
        print(self.pos)


def _eval_once(saver, summary_writer, accuracy, summary_op, confusion_matrix_op):
    # def _eval_once(saver, summary_writer, accuracy, summary_op, confusion_matrix_op, logits, labels, dense_labels):

    """Runs Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_1_op: Top 1 op.
      top_5_op: Top 5 op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if CKPT_PATH is not None:
            saver.restore(sess, CKPT_PATH)
            global_step = CKPT_PATH.split('/')[-1].split('-')[-1]
            print('Succesfully loaded model from %s at step=%s.' %
                  (CKPT_PATH, global_step))
        elif ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            if os.path.isabs(ckpt.model_checkpoint_path):
                # Restores from checkpoint with absolute path.
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                # Restores from checkpoint with relative path.
                saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                                 ckpt.model_checkpoint_path))

            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/imagenet_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Succesfully loaded model from %s at step=%s.' %
                  (ckpt.model_checkpoint_path, global_step))
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / BATCH_SIZE))
            # Counts the number of correct predictions.
            total_correct_count = 0.0
            total_false_positive_count = 0.0
            total_false_negative_count = 0.0
            total_sample_count = num_iter * BATCH_SIZE
            step = 0

            print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
            start_time = time.time()
            while step < num_iter and not coord.should_stop():
                correct_count, confusion_matrix = \
                    sess.run([accuracy, confusion_matrix_op])

                # correct_count, confusion_matrix, logits_v, labels_v, dense_labels_v = \
                #     sess.run([accuracy, confusion_matrix_op, logits, labels, dense_labels])

                total_correct_count += np.sum(correct_count)
                # total_false_positive_count += false_positive
                # total_false_negative_count += false_negative

                print('correct_count(step=%d): %d / %d' % (step, total_correct_count, BATCH_SIZE * (step + 1)))
                print('\nconfusion_matrix:')
                print(confusion_matrix)
                step += 1
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = BATCH_SIZE / sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                          'sec/batch)' % (datetime.now(), step, num_iter,
                                          examples_per_sec, sec_per_batch))
                    start_time = time.time()

            # print('total_false_positive_count: %d' % total_false_positive_count)
            # print('total_false_negative_count: %d' % total_false_negative_count)
            # Compute precision @ 1.
            precision = total_correct_count / total_sample_count
            print('%s: precision = %.4f [%d examples]' %
                  (datetime.now(), precision, total_sample_count))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision', simple_value=precision)
            summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def calc_metrics(dense_labels, logits):
    print("Precision", sk.metrics.precision_score(dense_labels, logits))
    print("Recall", sk.metrics.recall_score(dense_labels, logits))
    print("f1_score", sk.metrics.f1_score(dense_labels, logits))
    print("confusion_matrix")
    print(sk.metrics.confusion_matrix(dense_labels, logits))


def evaluate(dataset):
    """Evaluate model on Dataset for a number of steps."""
    # with tf.Graph().as_default():
    #     # Get images and labels from the dataset.
    #     images, labels = inputs(dataset, BATCH_SIZE)
    #
    #     # Number of classes in the Dataset label set plus 1.
    #     # Label 0 is reserved for an (unused) background class.
    #     num_classes = dataset.num_classes()
    #
    #     # Build a Graph that computes the logits predictions from the
    #     # inference model.
    #     logits, _ = inception.inference(images, num_classes)
    #
    #     sparse_labels = tf.reshape(labels, [BATCH_SIZE, 1])
    #     indices = tf.reshape(tf.range(BATCH_SIZE), [BATCH_SIZE, 1])
    #     concated = tf.concat(1, [indices, sparse_labels])
    #     num_classes = logits[0].get_shape()[-1].value
    #     dense_labels = tf.sparse_to_dense(concated,
    #                                       [BATCH_SIZE, num_classes],
    #                                       1, 0)
    #
    #     confusion_matrix_op = metrics.confusion_matrix(labels, tf.argmax(logits, axis=1))
    #     # false_positive_op = metrics.streaming_false_positives(logits, dense_labels)
    #     # false_negative_op = metrics.streaming_false_negatives(logits, dense_labels)
    #
    #     # Calculate predictions.
    #     accuracy = tf.nn.in_top_k(logits, labels, 1)
    #
    #     # Restore the moving average version of the learned variables for eval.
    #     variable_averages = tf.train.ExponentialMovingAverage(
    #         inception.MOVING_AVERAGE_DECAY)
    #     variables_to_restore = variable_averages.variables_to_restore()
    #     saver = tf.train.Saver(variables_to_restore)
    #
    #     # Build the summary operation based on the TF collection of Summaries.
    #     summary_op = tf.summary.merge_all()
    #
    #     graph_def = tf.get_default_graph().as_graph_def()
    #     summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, graph_def=graph_def)
    #
    #     while True:
    #         # _eval_once(saver, summary_writer, accuracy, summary_op, confusion_matrix_op,
    #                                                          logits, labels, dense_labels)
    #
    #         _eval_once(saver, summary_writer, accuracy, summary_op, confusion_matrix_op)
    #         if FLAGS.run_once:
    #             break
    #         time.sleep(FLAGS.eval_interval_secs)


def extract_patch_from_bb(bounding_box, wsi_image, level_used):
    # factor to map low res cords into high res
    mag_factor = pow(2, level_used)
    b_x_start = int(bounding_box[0]) * mag_factor
    b_y_start = int(bounding_box[1]) * mag_factor
    b_x_end = (int(bounding_box[0]) + int(bounding_box[2])) * mag_factor
    b_y_end = (int(bounding_box[1]) + int(bounding_box[3])) * mag_factor
    x_cords = np.arange(b_x_start, b_x_end, mag_factor)
    y_cords = np.arange(b_y_start, b_y_end, mag_factor)
    print(len(x_cords))
    print(len(y_cords))
    for y in y_cords:
        for x in x_cords:
            wsi_patch = wsi_image.read_region((x, y), 0, (data.PATCH_SIZE, data.PATCH_SIZE))
            patch = np.array(wsi_patch)
            print('processing: (%d, %d)' % (x, y))
            patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            lower_red = np.array([20, 20, 20])
            upper_red = np.array([200, 200, 200])
            mask = cv2.inRange(patch_hsv, lower_red, upper_red)
            white_pixel_cnt = cv2.countNonZero(mask)

            if white_pixel_cnt > ((data.PATCH_SIZE * data.PATCH_SIZE) * 0.50):
                print('*******************  accepted *****************************')
                patch = patch[:, :, :3]
                patch = np.divide(patch, 255)

                # subtract channel wise mean
                mean = np.mean(patch, axis=(0, 1))
                mean = np.reshape(mean, (1, 1, 3))
                patch = np.subtract(patch, mean)

                patch_queue.put(patch, (x, y))

            wsi_patch.close()


def build_heat_map(image_path, mask_path):
    wsi_image, rgb_image, level_used = wsi_ops.read_wsi_tumor(image_path, mask_path)
    if wsi_image is None:
        return

    bounding_boxes = wsi_ops.find_roi_bb_tumor(rgb_image)
    print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))
    for bbox in bounding_boxes[:1]:
        extract_patch_from_bb(bbox, wsi_image, level_used)

    patch_queue.print()


if __name__ == '__main__':
    # dataset = Dataset(DATA_SET_NAME, data_subset[1])
    # evaluate(dataset)
    wsi_ops = WSIOps()
    wsi_image_names = glob.glob(os.path.join(data.TUMOR_WSI_PATH, '*.tif'))
    wsi_image_names.sort()
    wsi_mask_names = glob.glob(os.path.join(data.TUMOR_MASK_PATH, '*.tif'))
    wsi_mask_names.sort()

    image_mask_pair = zip(wsi_image_names, wsi_mask_names)
    image_mask_pair = list(image_mask_pair)
    image_mask_pair = image_mask_pair[len(image_mask_pair)-1:]
    patch_queue = Queue()
    for image_name, mask_name in image_mask_pair:
        build_heat_map(image_name, mask_name)
