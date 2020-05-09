#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""Main scope of the program,
....
"""


from MachineLearning import Trainer,SerializedTrainer,starting_procedure_GM12878,tm_routine,evaluate_metrics
import sys
import Config
import cPickle
import logging
from sklearn import metrics
from sklearn import warnings
import copy
import pandas as pd
from sklearn.linear_model import LogisticRegression
import DataLoader


import os
import sys





def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[1]))


if __name__ == "__main__":
 with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  Config.load(sys.argv[1])
  os.chdir(os.path.dirname(os.path.abspath(sys.argv[1])))

  logging.basicConfig(format='%(asctime)s %(message)s',filename=Config.D["LOGFILE"], level=logging.DEBUG)

  #print get_script_path()
  #print os.getcwd()

  #os.chdir(get_script_path())

  #load configuration file
  #configuration is stored in global dictionary Config.D
  logging.info("Loaded configuration file: {0}".format(sys.argv[1]))

  sc=DataLoader.Data.create_from_file(Config.D["SC"], "SC", exclude=["GenTrain Score"])
  logging.info("Loaded " + Config.D["SC"])
  print "Loaded " + Config.D["SC"]

  #remove gdna samples, note that this needs to be changed based on the gdna nomenclature
  #for purely single cell dataset this should do nothing
  sc=sc.remove("gdna")
  logging.info("removed gDNA from SC dataset if present" )


  #apply threshodl, by default this is 0.01 and invalidates only very low quality calls from the input dataset
  sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=True)
  logging.info("Applied threshold {}".format(Config.D["TRAINING_SCORE_THRESHOLD"]))
  print "Applied threshold {}".format(Config.D["TRAINING_SCORE_THRESHOLD"])

  #restrict only to chromosomes defined in the configuration file
  sc.restrict_chromosomes(Config.D["CHROMOSOMES"])
  logging.info("Selected chromosomes based on cofig file...")

  #apply MA transformation
  sc.calculate_transformations_2()



  with open(Config.D["CLASSIFIER"], "rb") as input_file:
      #LOAD serialized Random Forest
      clf= SerializedTrainer(cPickle.load(input_file))
      logging.info("Loaded classifier {}".format(Config.D["CLASSIFIER"]))
      print "Loaded classifier {}".format(Config.D["CLASSIFIER"])
      #################################
      #apply predictions and merge it with the original dataframe sc
      sc=clf.predict_decorate(sc)
      logging.info("Loaded classifier {}".format(Config.D["CLASSIFIER"]))
      print "Applied prediction of {}".format(Config.D["CLFNAME"])
      #################################

      #initialize config to 2nd layer - GDA
      Config.D["OUTPUT_FEATURE"]="{0}_pred".format(Config.D["CLFNAME"])
      Config.D["CLFNAME"]=Config.D["CLFNAME2"]

      #split dataset into autosomes and sex-chromosomes
      chrom_l=dict()
      for (autosomes,sc_l) in sc.loop_chromosomes_autosomes_XY():
             if autosomes: chrom_l["AUTOSOMES"]=sc_l
             else: chrom_l["SEX"]=sc_l

      FILEACS=False



      with open(Config.D["OUTPUTDATAFILE"], 'w') as csvf, open(Config.D["GENOMESTUDIOFILE"],'w') as gs:
        for i in chrom_l:
          logging.info("Model {0}".format(str(i)))
          #vbclf=Trainer(chrom_l[i][0],jobid=0)
          logging.info("Fitting GDA model to group  {}".format(i))
          print "Fitting GDA model to group  {}".format(i)
          vbclf=Trainer(chrom_l[i],jobid=0)
          vbclf.train()
          chrom_l[i]=vbclf.predict_decorate(chrom_l[i])
          if not FILEACS:
             chrom_l[i].save_genome_studio_import_table(gs,header=True)
             chrom_l[i].save_complete_table(csvf,header=True)
             FILEACS=True
          else:
             chrom_l[i].save_genome_studio_import_table(gs,header=False)
             chrom_l[i].save_complete_table(csvf,header=False)







