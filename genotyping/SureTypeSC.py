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
import copy
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os


#load configuration file
Config.load(sys.argv[1])

OUTPUT_VARS=["m",
             "a",
             "x",
             "y",
             #"x_raw",
             #"y_raw",
             #"m_raw",
             #"a_raw",
                   "output",
                   "gtype",
                   "{0}_pred".format(Config.D["CLFNAME"]),
                   "{0}_pred".format(Config.D["CLFNAME2"]),
                   "{0}_prob".format(Config.D["CLFNAME"]),
                   "{0}_prob".format(Config.D["CLFNAME2"]),
                   "score"]

CLASIFIERS_VARS=["{0}_prob".format(Config.D["CLFNAME"]),
                       "{0}_prob".format(Config.D["CLFNAME2"]),
                       "score"]


THRESHOLDS=[0.5,
                  0.9,
                  0.15]


#LOGREGCOL="{0}_logreg".format(Config.D["CLFNAME2"])
#VBGMMCLFCOL="{0}_prob".format(Config.D["CLFNAME2"])


HEADER=True

os.chdir(os.path.dirname(os.path.abspath(sys.argv[1])))

logging.basicConfig(format='%(asctime)s %(message)s',filename=Config.D["LOGFILE"], level=logging.DEBUG)
logging.info("Loaded configuration file: " + sys.argv[1])

sc,ref=starting_procedure_GM12878()
logging.info("Loaded  testing data {0}".format(Config.D["SC"]) )

#load first level of classifier
if not Config.D["CROSSVAL"]:
  with open(Config.D["CLASSIFIER"], "rb") as input_file:

      clf= SerializedTrainer(cPickle.load(input_file))

      logging.info("Loaded configuration file: {0}".format(sys.argv[1]))





      sc=clf.predict_decorate(sc)
      logging.info("Decorated with prediction from : " + Config.D["CLASSIFIER"])

      ############

      ##############
      #eventually output some statistics here? call rates, adi ado rates...
      #############

      ##################################################################
      #Create the Variational Bayes model
      ##################################################################

      Config.D["OUTPUT_FEATURE"]="{0}_pred".format(Config.D["CLFNAME"])
      Config.D["CLFNAME"]=Config.D["CLFNAME2"]


      chrom_l=dict()
      transition_matrix_container=dict()



      if 1:
         #for chrom,sc_l in sc.loop_chromosomes():
         for (chrom,sc_l),(chrom_r,ref_l) in zip(sc.loop_chromosomes(),ref.loop_chromosomes()):
              chrom_l[chrom]=(sc_l,ref_l)
         for (autosomes,sc_l),(autosomes_r,ref_l) in zip(sc.loop_chromosomes_autosomes_XY(),ref.loop_chromosomes_autosomes_XY()):
             if autosomes: chrom_l["AUTOSOMES"]=(sc_l,ref_l)
             else: chrom_l["SEX"]=(sc_l,ref_l)



         #dump results of per chromosomes analysis
         with open(Config.D["TABLE"], 'w') as csvf:
            #loop over chromosomes defined in the config file, create VBgmm for every chromosome
            #for chrom,sc_l in sc.loop_chromosomes():

            for i in chrom_l:
                logging.info("Model {0}".format(str(i)) )
                vbclf=Trainer(chrom_l[i][0],jobid=0)
                vbclf.train()
                print i
                chrom_l[i]=(vbclf.predict_decorate(chrom_l[i][0]),chrom_l[i][1])

                data_test=chrom_l[i][0].df.stack(level=0)[OUTPUT_VARS].dropna()

                data_test["MODEL"]=i


                if HEADER:
                    data_test.to_csv(Config.D["CSV"],sep=",",header=True,mode="w")
                    HEADER=False
                else: data_test.to_csv(Config.D["CSV"],sep=",",header=False,mode="a")

                loop=[("het",dfg) if g else ("homo",dfg)  for g,dfg in data_test.groupby(data_test["gtype"]=="AB")] + [('all',data_test)]

                for g,dfg in loop:
                  for func,func_name in [#(classification_report,"clf report"),
                                   (metrics.precision_score,"precision"),
                                   (metrics.accuracy_score,"accuracy"),
                                   (metrics.recall_score,"recall"),
                                   (metrics.roc_auc_score,"roc_auc_score"),
                                   (metrics.f1_score,"f1_score"),
                                   (metrics.average_precision_score,"average_prec_score")]:

                     for clf,thr in zip(CLASIFIERS_VARS,THRESHOLDS):
                         if  func_name!="roc_auc_score":
                           outps="{0},{1},{2},{3},{4},{5}".format(i,
                                                          func_name,
                                                          func(dfg["output"],dfg[clf]>=thr),
                                                          clf,
                                                          thr,
                                                          g)
                         else:
                             print dfg["output"]
                             outps="{0},{1},{2},{3},{4},{5}".format(i,
                                                          func_name,
                                                          func(dfg["output"],dfg[clf]),
                                                          clf,
                                                          thr,
                                                          g)
                         #print outps
                         csvf.write(outps+"\n")




                tm_routine(transition_matrix_container,chrom_l[i][1],chrom_l[i][0],Config.D["TRAINING_SCORE_THRESHOLD"],i,"unfiltered")

                stack_sc=[copy.deepcopy(chrom_l[i][0]),copy.deepcopy(chrom_l[i][0]),copy.deepcopy(chrom_l[i][0]),copy.deepcopy(chrom_l[i][0])]
                #CLASIFIERS_VARS
                #THRESHOLDS

                for sc_sp,clf,thr in zip(stack_sc,CLASIFIERS_VARS + ["score+vbgmm"],THRESHOLDS + [0.15]):
                  #sc_sp.apply_proba_prediction_results(clf,thr)
                  if clf=="score+vbgmm":
                      sc_sp.apply_proba_prediction_results(clfname="score",threshold=0.15)
                      sc_sp.apply_proba_prediction_results(clfname="{0}_prob".format(Config.D["CLFNAME2"],threshold=0.9))
                      tm_routine(transition_matrix_container,chrom_l[i][1],sc_sp,thr,i,clf)

                  else:
                    sc_sp.apply_proba_prediction_results(clfname=clf,threshold=thr)
                    tm_routine(transition_matrix_container,chrom_l[i][1],sc_sp,thr,i,clf)


            pd.concat(transition_matrix_container,axis=0).to_csv(Config.D["MATRIX"],sep=";")


else:
  ar=[]
  layer1=Config.D["CLFNAME"]
  layer1_output_feat=Config.D["OUTPUT_FEATURE"]
  layer2=Config.D["CLFNAME2"]
  layer2_output_feat="{0}_pred".format(layer1)
  for chrom,sc_l in  sc.loop_chromosomes_autosomes_XY():
    for runnr,train,test in sc_l.stratify(10,revert=False):
      ####1st layer
      Config.D["CLFNAME"]=layer1
      Config.D["OUTPUT_FEATURE"]=layer1_output_feat
      training=Trainer(train,jobid=str(runnr))
      training.train()
      test = training.predict_decorate(test)
      ###2nd layer
      Config.D["CLFNAME"]=layer2
      Config.D["OUTPUT_FEATURE"]=layer2_output_feat
      vbclf=Trainer(test,jobid=str(runnr))
      vbclf.train()
      test=vbclf.predict_decorate(test)

      #now evaluate
      for clfvar,thr in zip(CLASIFIERS_VARS,THRESHOLDS):
         for res in evaluate_metrics(test.df,
                                    thr=thr,
                                    colname=clfvar,
                                    names=["precision","recall","f1score","accuracy"],
                                    func=[metrics.precision_score,metrics.recall_score,metrics.f1_score,metrics.accuracy_score]):
            ar.append(tuple([runnr,clfvar]  + list(res) + [chrom]))



  cf_df=pd.DataFrame.from_records(ar,columns=["run","algorithm","metrics","val","autosomes"])
  #print cf_df
  cf_df.to_csv(Config.D["CROSSFOLDOUT"])

