# Copyright 2015 Google Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow.python.platform

import collections
import math
import numpy as np

import os
import time

import random
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

import input_data
import NNClassifier as nn

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')

flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('verbose_steps', 200, 'Number of steps to return training loss.')
flags.DEFINE_integer('checkpoint_steps', 1000, 'Number of steps to create a checkpoint.')

flags.DEFINE_integer('inputDim', 14, 'Input dimension') # 28*28 for MNIST example
flags.DEFINE_integer('num_classes', 12, 'Number of output classes (dimension)')

flags.DEFINE_integer('hidden1', 20, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 20, 'Number of units in hidden layer 2.')

flags.DEFINE_integer('batch_size', 10, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')

flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    input_placeholder: Inputs placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  input_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         FLAGS.inputDim))
  labels_placeholder = tf.placeholder(tf.int32, shape=(None)) # None, 1

  return input_placeholder, labels_placeholder

def fill_feed_dict(data_set, inputs_pl, labels_pl):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of inputs and labels, from input_data.read_data_sets()
    inputs_pl: The inputs placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  inputs_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)

  feed_dict = {
      inputs_pl: inputs_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

def output_test(sess, label, inputs_pl, test_set):
  """Run one round on the output, and optionally output to a csv
  """
  print("Outputing Test")
  
  "Debug test with small sample size"

  feed_dict = {
    inputs_pl : test_set,
  }

  #outputArray = label.eval(feed_dict=feed_dict, session=sess)

  outputArray = sess.run(label, feed_dict=feed_dict) # this seems not working
  return outputArray

def do_eval(sess,
            eval_correct,
            inputs_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    inputs_placeholder: The inputs placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of inputs and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               inputs_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

def run_training():
  """Train classifier for a number of steps."""
  # Get the sets of inputs and labels for training, validation
  data_sets = input_data.read_data_sets()
  trainData = data_sets.trainData
  testData = data_sets.trainData.testInput

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the inputs and labels.
    inputs_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = nn.inference(inputs_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2,
                             FLAGS.num_classes)

    # Add label to the Graph the Ops for feed forward output.
    label = nn.output(logits)

    # Add to the Graph the Ops for loss calculation.
    loss = nn.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = nn.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = nn.evaluation(logits, labels_placeholder)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                            graph_def=sess.graph_def)

    # And then after everything is built, start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of inputs and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(trainData,
                                 inputs_placeholder,
                                 labels_placeholder)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % FLAGS.verbose_steps == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % FLAGS.checkpoint_steps == 0 or (step + 1) == FLAGS.max_steps:
        saver.save(sess, FLAGS.train_dir, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                inputs_placeholder,
                labels_placeholder,
                trainData)

        # Evaluate against the validation set.
        # print('Validation Data Eval:')
        # do_eval(sess,
        #         eval_correct,
        #         inputs_placeholder,
        #         labels_placeholder,
        #         data_sets.validation)
        # # Evaluate against the test set.
        # print('Test Data Eval:')
        # do_eval(sess,
        #         eval_correct,
        #         inputs_placeholder,
        #         labels_placeholder,
        #         data_sets.test)

    # At max step
    outputArray = output_test(sess,
                label,
                inputs_placeholder,
                testData)

    input_data.write_csv('testOutput.csv', outputArray)
        

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()