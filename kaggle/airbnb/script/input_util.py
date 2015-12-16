# Input parser for Airbnb data in csv
# ==============================================================================

from __future__ import absolute_import

import gzip
import os

import numpy as np
from dateutil.parser import parse

# TODO, making it more robust
def num(s, default):
    try:
      return float(s)
    except ValueError:
      return default

def convertToNum(nparray, columnList, default = 0.0):
  npSubArray = nparray[:,columnList]
  g = np.vectorize(num)
  nparray[:,columnList] = g(npSubArray, default)

  # for (x, y), val in np.ndenumerate(npSubArray):
  #   npSubArray[x,y] = num(val, default)


  return nparray

'''
  delta type supports: Day, Sec
'''
def parseDateIntoDelta(dateStr, baseDate, default, deltaType):
  try:
    delta = parse(dateStr) - baseDate
    if deltaType == 'Day':
      return delta.days
    else:
      return delta.seconds
  except ValueError:
    return default

'''
  convert date into date diff
'''
def convertDate(nparray, columnList, baseDateStr = '1/1/2010', default = -1, deltaType = 'Day'):
  npSubArray = nparray[:,columnList]
  # lambda implementation here
  baseDate = parse(baseDateStr)
  g = np.vectorize(parseDateIntoDelta)
  nparray[:,columnList] = g(npSubArray,baseDate,default, deltaType)
  return nparray

# inplace normalization
def columnNormalizer(nparray, columnList, minVal = 0, maxVal = 10):
  npSubArray = nparray[:,columnList].astype(float, copy = False)
  
  npSubArrayMin = npSubArray.min(axis=0)
  npSubArrayMax = npSubArray.max(axis=0)
  
  npSubArray -= npSubArrayMin
  npSubArray *= maxVal - minVal
  npSubArray /= npSubArrayMax - npSubArrayMin
  npSubArray += minVal

  nparray[:,columnList] = npSubArray
  return nparray

def tokenizeList(nparray):
  str2ind_column = {}
  ind2str_column = {}
  cnt = 0
  for x, val in np.ndenumerate(nparray):
    if val not in str2ind_column:
      str2ind_column[val] = cnt
      ind2str_column[cnt] = val
      cnt += 1
    nparray[x] = str2ind_column[val]
  return str2ind_column, ind2str_column

'''
  columnList: columns that are getting tokenized
'''
def tokenizeByColumn(npdatatable, columnList, name = None):
  
  str2ind = []
  ind2str = []

  for c in columnList:
    str2ind_column, ind2str_column = tokenizeList(npdatatable[:,c])
    str2ind.append(str2ind_column)
    ind2str.append(ind2str_column)

  if name is not None:
    print name, str2ind

  return str2ind, ind2str



