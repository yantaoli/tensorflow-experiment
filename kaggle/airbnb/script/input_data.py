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

    for x in range(data.get_shape().as_list()[0]): 
    #for x in range(data.size):
      writer.writerow([data[x]])

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

# TODO support test datasets
class UserDataSet(object):
  def __init__(self, train_users, train_users_header, test_set, test_set_header,one_hot=False):
    self._train_users = train_users
    self._train_users_header = train_users_header

    self._num_examples = train_users.shape[0]

    self.train_users_str2ind, self.train_users_ind2str = input_util.tokenizeByColumn(train_users,[4,6,8,9,10,11,12,13,14,15])
    input_util.convertToNum(train_users, [2,5,7], default = 0.0)
    input_util.convertDate(train_users, [1,3], baseDateStr = '1/1/2010', default = -1)
    input_util.columnNormalizer(train_users, [1,3], minVal = 0.0, maxVal = 10.0)
    input_util.columnNormalizer(train_users, [2,5,7], minVal = 0.0, maxVal = 10.0)


    self._input = train_users[:, range(1,15)].astype(float)
    self._output = train_users[:, 15].astype(int)

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

  TRAIN_USERS = 'train_users.csv'
  TEST_USERS = 'test_users.csv'
  #SESSIONS = 'sessions.csv' # comment out because sessions data is too big to fit in python memory
  COUNTRIES = 'countries.csv'
  AGE_GENDER_BKTS = 'age_gender_bkts.csv'

  train_users, train_header= extract_csv(TRAIN_USERS)
  test_users, test_headers = extract_csv(TEST_USERS)
  # process training data
  data_sets.trainData = UserDataSet(train_users, train_header, test_users, test_headers)


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