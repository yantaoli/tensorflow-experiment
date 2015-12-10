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

def tokenizeList(nparray):
  str2ind_column = {}
  ind2str_column = {}
  cnt = 0
  for val in nparray:
    if val not in str2ind_column:
      str2ind_column[val] = cnt
      ind2str_column[cnt] = val
      cnt += 1

  return str2ind_column, ind2str_column

def tokenizeByColumn(npdatatable):
  row, column = datatable.shape
  
  str2ind = []
  ind2str = []

  for c in range(column):
    str2ind_column, ind2str_column = tokenizeList(npdatatable[:,c])
    str2ind.append(str2ind_column)
    ind2str.append(ind2str_column)

  return str2ind, ind2str



