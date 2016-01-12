# Input parser for Airbnb data in csv
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy as np

import csv
import input_util

PATH = '../data/'

def write_csv(filename, data):
  with open(PATH + filename, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for x in range(data.shape[0]): 
    #for x in range(data.size):
      # debug
      # print(data[x])
      writer.writerow([data[x]]) # it is 1D array, thus no need to add [x, :].tolist()

  print('CSV data written' + PATH + filename)


def extract_csv(filename, hasHeader = True):
  with open(PATH + filename, 'rb') as csvfile:
    cvsreader = csv.reader(csvfile, delimiter=',')
    isHeaderSet = not hasHeader
    header = []
    data = []
    for row in cvsreader:
      if not isHeaderSet:
        header = row
        isHeaderSet = True
      else:
        data.append(row)
    return np.array(data), header

# support test datasets
class UserDataSet(object):
  def __init__(self, training_set, training_set_header, test_set, test_set_header):
    self._training_set = training_set
    self._training_set_header = training_set_header
    self._test_set = test_set

    self._num_examples = training_set.shape[0]
    self._total_dim = training_set.shape[1]
    self._num_tests = test_set.shape[0]
    self._test_total_dim = test_set.shape[1]

    print("Training Samples:")
    print(self._num_examples)
    
    print("Testing Samples")
    print(self._num_tests)

    # Define set parameters
    self._input_dim = 15 # column 0 to 14
    self._output_dim = 1
    self._null_value = -99

    # Augment test set to the same size as training set
    assert self._test_total_dim <= self._total_dim
    if self._test_total_dim < self._total_dim:
      # resources at: http://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-an-numpy-array
      numColumnToAppend = self._total_dim - self._test_total_dim
      augmentedTestSet = np.c_[test_set, np.ones((self._num_tests,numColumnToAppend)) * self._null_value]
    else:
      # to assert test dim should never be 
      augmentedTestSet = test_set

    # Augment the sets
    self._augmentedSet = np.r_[self._training_set ,augmentedTestSet]
    self._total_sample = self._augmentedSet.shape[0]
    print("Augmented Samples:")
    print(self._total_sample)

    # tokenize + normalize
    self.str2ind, self.ind2str = input_util.tokenizeByColumn(self._augmentedSet,[4,6,8,9,10,11,12,13,14,15])
    input_util.convertToNum(self._augmentedSet, [2,5,7], default = 0.0)
    input_util.convertDate(self._augmentedSet, [1,3], baseDateStr = '1/1/2010', default = -1)
    input_util.columnNormalizer(self._augmentedSet, [1,3], minVal = 0.0, maxVal = 10.0)
    input_util.columnNormalizer(self._augmentedSet, [2,5,7], minVal = 0.0, maxVal = 10.0)

    inputsArray = self._augmentedSet[:, range(1,15)].astype(float)
    outputsArray = self._augmentedSet[:, 15].astype(int)

    self._input = inputsArray[range(0,self._num_examples), :]
    self._testInput = inputsArray[range(self._num_examples, self._total_sample), :]

    self._output = outputsArray[range(0,self._num_examples)] # the data is 1D

    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def input(self):
    return self._input

  @property
  def output(self):
    return self._output

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  @property
  def testInput(self):
    return self._testInput

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)

      self._input = self._input[perm]
      self._output = self._output[perm]

      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples

    end = self._index_in_epoch
    return self._input[start:end], self._output[start:end]



def read_data_sets():
  class DataSets(object):
    pass
  data_sets = DataSets()

  TRAINING_SET = 'train_users.csv'
  TEST_USERS = 'test_users.csv'
  #SESSIONS = 'sessions.csv' # comment out because sessions data is too big to fit in python memory
  COUNTRIES = 'countries.csv'
  AGE_GENDER_BKTS = 'age_gender_bkts.csv'

  training_set, train_header= extract_csv(TRAINING_SET)
  test_users, test_headers = extract_csv(TEST_USERS)

  # TODO append test users after train users then perform normalizations


  # process training data
  data_sets.trainData = UserDataSet(training_set, train_header, test_users, test_headers)


  # data_sets.sessions, _ = extract_csv(SESSIONS)
  data_sets.countries, _ = extract_csv(COUNTRIES)
  data_sets.age_gender_bkts, _ = extract_csv(AGE_GENDER_BKTS)
  
  return data_sets

def read_data_set_session():
  SESSIONS = 'sessions.csv'
  sessions, _ = extract_csv(SESSIONS)
  return sessions


# testing function
def read():
  return read_data_sets()