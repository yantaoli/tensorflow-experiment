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

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.

TensorFlow install instructions:
https://tensorflow.org/get_started/os_setup.html

MNIST tutorial:
https://tensorflow.org/tutorials/mnist/tf/index.html
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow.python.platform
import tensorflow as tf

print("Test 1")
# Build a dataflow graph.
c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d)

# Construct a `Session` to execut the graph.
sess = tf.InteractiveSession()

# Execute the graph and store the value that `e` represents in `result`.
result = sess.run(e)
print(result)
sess.close()

print("Test 2")
sess = tf.InteractiveSession()
# test 2, output from graph
input_placeholder = tf.placeholder(tf.float32, shape=(3,2))
biases = tf.constant(tf.zeros([2]),
hidden1 = input_placeholder + biases
output = tf.argmax(input_placeholder, 0, name='output')

test_set_sample = np.array([[1,2],
                            [3,2],
                            [3,4]])
print(test_set_sample)

feed_dict = {
  input_placeholder : test_set_sample,
}

#outputArray = label.eval(feed_dict=feed_dict, session=sess)
outputArray1 = sess.run(hidden1, feed_dict=feed_dict) # this seems not working
outputArray2 = sess.run(output, feed_dict=feed_dict) # this seems not working
print(outputArray1)
print(outputArray2)

sess.close()