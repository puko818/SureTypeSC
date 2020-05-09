import numpy as np
import pandas as pd

import math


def M(x): return np.log2(x[0]) - np.log2(x[1])
#bug 0.5*0.5!!! 2.10. 2017
#def A(x): return 0.5 * 0.5 * (np.log2(x[0]) + np.log2(x[1]))
def A(x): return 0.5 * (np.log2(x[0]) + np.log2(x[1]))

def M_2(x,y):
    return np.log2(x.astype(float)) - np.log2(y.astype(float))
#bug 0.5*0.5!!! 2.10. 2017
#def A_2(x,y): return 0.5 * 0.5 * (np.log2(x) + np.log2(y))
def A_2(x,y): return 0.5 * (np.log2(x.astype(float)) + np.log2(y.astype(float)))

def Equal(x): return x[0]==x[1]
def Equal_with_Nan(x):
    #if np.isnan(x[0]) or np.isnan(x[1]):
    #if np.math.isnan(x[0]) or np.math.isnan(x[1]):
    #if np.isnan(x[0]) or np.isnan(x[1]):
    if pd.isnull(x[0]) or pd.isnull(x[1]):
        return np.nan
    else:
        return x[0]==x[1]

def Equal_with_Nan_and_Categories(x):
    if pd.isnull(x[0]) or pd.isnull(x[1]):
        return np.nan
    else:
        suf=str(x[0]==x[1])
        return str(x[0])+ "_" + suf


def Event_with_Nan(x):
  retval="NORMAL"
  if pd.isnull(x[0]) or pd.isnull(x[1]) or x[0]=='NC' or x[1]=='NC': retval=np.nan
  elif (x[0]=='AA' or x[0]=='BB') and  (x[1]=='AB'):
          retval="ADI"
  elif (x[0]=='AB') and (x[1]=='AA' or x[1]=='BB'):
          retval="ADO"

  return retval

def Identity(x):
    return x[1]




def Homozygous(x):
    pass



def Heterozygous(x):
    pass