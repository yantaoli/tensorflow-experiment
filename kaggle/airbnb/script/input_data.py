# Input parser for Airbnb data in csv
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import csv

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
    return data, header

def read_data_sets():
  class DataSets(object):
    pass
  data_sets = DataSets()

  TRAIN_USERS = 'train_users.csv'
  TEST_USERS = 'test_users.csv'
  SESSIONS = 'sessions.csv'
  COUNTRIES = 'countries.csv'
  AGE_GENDER_BKTS = 'age_gender_bkts.csv'

  data_sets.train_users, _ = extract_csv(TRAIN_USERS)
  data_sets.test_users, _ = extract_csv(TEST_USERS)
  data_sets.sessions, _ = extract_csv(SESSIONS)
  data_sets.countries, _ = extract_csv(COUNTRIES)
  data_sets.age_gender_bkts, _ = extract_csv(AGE_GENDER_BKTS)

  return data_sets

# testing function
def test():
  data_sets = read_data_sets()