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
import input_tokenization

PATH = '../data/'

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
      data.append(row)
    return np.array(data), header

def read_data_sets():
  class DataSets(object):
    pass
  data_sets = DataSets()

  TRAIN_USERS = 'train_users.csv'
  TEST_USERS = 'test_users.csv'
  #SESSIONS = 'sessions.csv' # comment out because sessions data is too big to fit in python memory
  COUNTRIES = 'countries.csv'
  AGE_GENDER_BKTS = 'age_gender_bkts.csv'

  data_sets.train_users, _ = extract_csv(TRAIN_USERS)
  data_sets.test_users, _ = extract_csv(TEST_USERS)
  #data_sets.sessions, _ = extract_csv(SESSIONS)
  data_sets.countries, _ = extract_csv(COUNTRIES)
  data_sets.age_gender_bkts, _ = extract_csv(AGE_GENDER_BKTS)

  # perform inplace tokenization
  data_sets.train_users_str2ind, data_sets.train_users_ind2str = input_tokenization.tokenizeByColumn(data_sets.train_users,[0,4,6,8,9,10,11,12,13,14,15],'train_users')

  return data_sets

def read_data_set_session():
  SESSIONS = 'sessions.csv'
  sessions, _ = extract_csv(SESSIONS)
  return sessions


# testing function
def test():
  data_sets = read_data_sets()