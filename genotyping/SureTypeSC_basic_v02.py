#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""Main scope of the program,
...Ivan Vogel
...
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


#First layer of classifiers, will be deserialized from a binary file
classifiers=["rf","gmmcl"]
classifiers2=["gmmcl"]



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

  #split dataset into autosomes and sex-chromosomes
  chrom_l=dict()
  for (autosomes,sc_l) in sc.loop_chromosomes_autosomes_XY():
             if autosomes: chrom_l["AUTOSOMES"]=sc_l
             else: chrom_l["SEX"]=sc_l


#open single serialized single layers
  for clf,clfname in zip(Config.D["CLASSIFIERS"],Config.D["CLFNAMES"]):
    with open(clf, "rb") as input_file:
      #LOAD classifier
      Config.D["CLASSIFIER"]=clf
      Config.D["CLFNAME"]=clfname
      clf= SerializedTrainer(cPickle.load(input_file))
      logging.info("Loaded classifier {}".format(Config.D["CLASSIFIER"]))
      print "Loaded classifier {}".format(Config.D["CLASSIFIER"])

      for i in chrom_l:
        chrom_l[i]=clf.predict_decorate(chrom_l[i],threshold=Config.D["CUTOFF"])
        logging.info("Finished prediction for MODEL: {},  {}, ratio: {}".format(i,Config.D["CLFNAME"],Config.D["RATIO"]))
      #################################
      #apply predictions and merge it with the original dataframe sc
      #sc=clf.predict_decorate(sc)
      #logging.info("Loaded classifier {}".format(Config.D["CLASSIFIER"]))
      #print "Applied prediction of {}".format(Config.D["CLFNAME"])
      #################################


#layer2_output_feat="{0}_pred".format(layer1)




      #initialize config to 2nd layer - GDA
      #Config.D["OUTPUT_FEATURE"]="{0}_pred".format(Config.D["CLFNAME"])
      #Config.D["CLFNAME"]=Config.D["CLFNAME2"]



      FILEACS=False


  
  with open(Config.D["OUTPUTDATAFILE"], 'w') as csvf, open(Config.D["GENOMESTUDIOFILE"],'w') as gs:
    for i in chrom_l:
       logging.info("Model {0}".format(str(i)) )
       first_layer="{}_ratio:{}".format(Config.D["CLFNAMES"][0],Config.D["RATIO"])
       Config.D["OUTPUT_FEATURE"]="{}_pred".format(first_layer)
    
       Config.D["CLFNAME"]=Config.D["CLFNAMES"][1]
       #fit the model from the 2nd layer
       train=Trainer(chrom_l[i])
       train.train()
       #create field for multiple layered classifier -> first layer + current classifier
       Config.D["CLFNAME"]="{}-{}".format(Config.D["CLFNAMES"][0],Config.D["CLFNAMES"][1])
       #apply the fitted model from the 2nd layer
       chrom_l[i]=train.predict_decorate(chrom_l[i])
       logging.info("Finished prediction for MODEL: {},  {}, ratio: {}".format(i,Config.D["CLFNAME"],Config.D["RATIO"]))
       if not FILEACS:
          chrom_l[i].save_genome_studio_import_table(gs,header=True)
          chrom_l[i].save_complete_table(csvf,header=True)
          FILEACS=True
       else:
          chrom_l[i].save_genome_studio_import_table(gs,header=False)
          chrom_l[i].save_complete_table(csvf,header=False)



  
  #output single cell genotypes in using mode defined in the configuration file, by default, all three modes are applied (standard genotyping, high precision and high recall)
  for m in Config.D["MODE"]:
   FILEACS=False
   outfilename="{}/{}_{}_{}_{}.txt".format(Config.D["OUTPUTDIRPATH"],Config.D["DATASET"],Config.D["ID"],"SC_genotypes",m)
   with open(outfilename,"w") as modefile: 
     for i in chrom_l:
     #if m=="precision":#precision mode: RF-GDA score 0.75
     #   classifier=Config.D["CLFNAME"]

      if m=="recall":# recall mode: RF score xyz
        classifier=Config.D["CLFNAMES"][0]
        threshold=0.15

      elif m=="precision":# standard mode: RF-GDA score 0.15
        classifier=Config.D["CLFNAME"]
        threshold=0.75
      else:
        classifier=Config.D["CLFNAME"]
        threshold=0.15

      dat=chrom_l[i].apply_threshold_generic(threshold,classifier)
      if not FILEACS:  
        dat.save_complete_table(modefile,header=True)
        FILEACS=True
      else:
        dat.save_complete_table(modefile,header=False)


