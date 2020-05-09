from collections import defaultdict
import pandas as pd
import Config
import sys
from tabulate import tabulate

_container=[]
_title=""
_dataname=""

_stat = defaultdict(list)



def init(t,d):
    global _title,_dataname
    _title=t
    _dataname=d


def send_data(_d):
    '''

    :return:
    '''
    _container.append(_d)


def add_stat(jobid,data):
    _stat[jobid].append(data)


def output():
    global _title,_dataname
    stdout_bk=sys.stdout
    with open(Config.D["PROTOCOL"],"w") as filename:
      sys.stdout = filename

      print _title
      print _dataname

      print "Path for SC:"
      print Config.D["SC"]

      print "Path for gDNA:",Config.D["GDNA"]

      print "Using chromosomes:",Config.D["CHROMOSOMES"]

      print "Threshold for gDNA consensus:",Config.D["THRESHOLD"]


      print "Threshold for training data:",Config.D["TRAINING_SCORE_THRESHOLD"]

      ##print "Threshold for testing data:",Config.D["SCORE_START"],Config.D["SCORE_END"],Config.D["SCORE_STEP"]


      if bool(_stat):
          print tabulate(pd.DataFrame.from_dict(_stat,orient='index'))
      for item in _container:
          print item


    sys.stdout=stdout_bk

