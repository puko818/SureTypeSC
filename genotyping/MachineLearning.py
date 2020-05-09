import os
import logging
from DataLoader import Data,Patterns

import Settings
import sys
from sklearn import mixture
import numpy as np
from collections import Counter
import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.covariance import MinCovDet
import datetime
import Config
#import pickle
import cPickle
#import TrioLoader
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import copy
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import copy
import random
from sklearn import tree
from sklearn import linear_model
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KNeighborsClassifier
from sklearn import mixture
import protocol
from sklearn import metrics
from sklearn import linear_model

from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import Imputer


from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GMM,BayesianGaussianMixture,GaussianMixture
from scipy.misc import logsumexp


#import joblib as jb


def test(x):
    return x

'''
functions create GMM classifier from dataframe
param d has cols:
"output" - true genotype (1 or 0)
"m" - logarithmic difference of signals
"a" - logarithmic average of signals
"gtype" - genotype
"pred" - prediction of 1st layer - usually random forest

returns GMM classifier
'''
'''
def _print(self, msg, msg_args):
    """ Display the message on stout or stderr depending on verbosity
    """
    # XXX: Not using the logger framework: need to
    # learn to use logger better.
    if not self.verbose:
        return
    if self.verbose < 50:
        writer = sys.stderr.write
    else:
        writer = sys.stdout.write
    msg = msg % msg_args
    writer('[%s]: %s\n' % (self, msg))
    logging.info('[%s]: %s\n' % (self, msg))
'''


def  create_components_for_gmm_roc(d,threshold):
    lookup="proba_" + Config.D["CLFNAME"]
    d=d[["gtype",lookup]+Config.D["INPUT_FEATURE"]]
    d[Config.D["INPUT_FEATURE"]]=d[Config.D["INPUT_FEATURE"]].replace(to_replace=[np.inf, -np.inf],value=np.nan)
    d.dropna(inplace=True)
    #d.loc[(~np.isfinite(d)) & d.notnull()] = np.nan

    #covariancemat=d.groupby(["gtype","pred"])["m","a"].cov()

    #positive_class=d[d[lookup] >= threshold]
    #negative_class=d[d[lookup] < threshold]

    positive_class=(d[d[lookup] >= threshold],1)
    negative_class=(d[d[lookup] < threshold],0)




    #means_positive=positive_class.groupby(["gtype"])[Config.D["INPUT_FEATURE"]].mean()
    #means_negative=negative_class.groupby(["gtype"])[Config.D["INPUT_FEATURE"]].mean()

    #this is really just to make it go through
    #if np.inf in means.values or -np.inf in means.values or np.nan in means.values:
    #   print "substituting 0 to means - this should not happen and would bias the analysis!"
    #   means=means.replace([np.inf, -np.inf], np.nan).replace(np.nan,0.0)

    #covariancemat.index=pd.MultiIndex.from_tuples([("_".join([str(i[0]),str(i[1])]),i[2]) for i in covariancemat.index.values],names=["component","feature"])

    #covariancemat.fillna(method='backfill',inplace=True)
    #covariancemat=covariancemat.swaplevel()

    #if np.inf in covariancemat.values or -np.inf in covariancemat.values or np.nan in covariancemat.values:
    #    covariancemat=covariancemat.replace([np.inf, -np.inf], np.nan).replace(np.nan,0.0)
    covariance_ar=[]
    weights_ar=[]
    means_ar=[]
    groups=[]
    #for i in range(covariancemat.to_panel().values.shape[1]):
    #    print np.linalg.pinv(covariancemat.to_panel().values[:,i])
    #    test_ar.append(np.linalg.pinv(covariancemat.to_panel().values[:,i]))



    ####THIS IS ORIGINAL VERSION

    for d_cl in [positive_class,negative_class]:
      for g,df in d_cl[0].groupby(["gtype"]):
            covariance_ar.append(np.linalg.pinv((df[Config.D["INPUT_FEATURE"]].cov()).values))
            weights_ar.append(len(df))
            means_ar.append(df[Config.D["INPUT_FEATURE"]].mean())
            groups.append((g,d_cl[1]))
        #return tuple of: (means,precision_matrix,weights)
    covariance_ar=np.array(covariance_ar)
    #covariance_ar[np.isnan(covariance_ar)]=1.0
    #previously we initialised everything
    #gmm=mixture.GaussianMixture(warm_start=True,n_components=6, covariance_type='full',means_init=means.values ,precisions_init=covariance_ar,weights_init=np.array([float(i)/sum(weights_ar) for i in weights_ar]))
    #gmm=mixture.GaussianMixture(n_components=6, precisions_init=covariance_ar,covariance_type='full',means_init=means_ar,weights_init=np.array([float(i)/sum(weights_ar) for i in weights_ar]),max_iter=10000,verbose=2)
    gmm=mixture.GaussianMixture(n_components=6, precisions_init=covariance_ar,covariance_type='full',means_init=means_ar,weights_init=np.array([float(i)/sum(weights_ar) for i in weights_ar]),max_iter=1000,verbose=0)
    #gmm=mixture.GaussianMixture(n_components=6,covariance_type='full',means_init=means_ar,weights_init=np.array([float(i)/sum(weights_ar) for i in weights_ar]),max_iter=10000,verbose=2)
    ##################################

    ##This is THE adjusted version with only 4 components
    '''
    for gpos,pos_df in positive_class.groupby(["gtype"]):
        covariance_ar.append(np.linalg.pinv((pos_df[Config.D["INPUT_FEATURE"]].cov()).values))
        means_ar.append(pos_df[Config.D["INPUT_FEATURE"]].mean())
        weights_ar.append(len(pos_df))
        groups.append(gpos)


    covariance_ar.append(np.linalg.pinv((negative_class[Config.D["INPUT_FEATURE"]].cov()).values))
    means_ar.append(negative_class[Config.D["INPUT_FEATURE"]].mean())
    weights_ar.append(len(negative_class))
    groups.append('Null')


    #gmm.means_=means_ar
    #gmm.precisions_=covariance_ar
    #gmm.weights_=np.array([float(i)/sum(weights_ar) for i in weights_ar])
    gmm=mixture.GaussianMixture(n_components=4, covariance_type='full',means_init=np.array(means_ar) ,precisions_init=np.array(covariance_ar),weights_init=np.array([float(i)/sum(weights_ar) for i in weights_ar]))
    '''

    return (groups,gmm)


class GMMcl:
    n_components = None
    covariance_type = None
    nClasses = None
    mixtures = None

    def __init__(self,core,n_components=None, covariance_type='full'):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.core=core


    def fit(self, X,Y):
        self.classes_=np.unique(Y)
        self.group_weights=[len(X[Y==i]) / float(len(X)) for i in self.classes_]
        #self.group_weights=self.groups.apply(len)/sum(self.groups.apply(len))
        self.group_log_weights=np.log(self.group_weights)
        self.nClasses = len(self.classes_)
        #self.mixtures = [BayesianGaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,max_iter=1000,verbose=2,verbose_interval=100)
        if self.core=="gmm":
          self.mixtures = [GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,max_iter=1000,verbose=0)
                         for k in range(self.nClasses)]
        else:
          self.mixtures = [BayesianGaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,max_iter=1000,verbose=0)
                         for k in range(self.nClasses)]
        #self.mixtures = [GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,max_iter=1000,verbose=2,verbose_interval=100)
        #                 for k in range(self.nClasses)]
        for g,mix in zip(self.classes_, self.mixtures):
           mix.fit(X[Y==g])
        #print self.mixtures.
        return self

    def predict(self, X):
        ll = [ mix.score_samples(X)for k, mix in enumerate(self.mixtures) ]
        #ll = [ mix.score(X[["m","a"]])for k, mix in enumerate(self.mixtures) ]
        #return np.vstack(ll).argmax(axis=0)
        return np.take(self.classes_, np.vstack(ll).argmax(axis=0), axis=0)

    def predict_proba(self,X):
        ll = [ mix.score_samples(X)for k, mix in enumerate(self.mixtures) ]
        #ll = [ mix.score(X[["m","a"]])for k, mix in enumerate(self.mixtures) ]
        #for ind,val in enumerate(self.groups.groups.keys()):
        #    if val==1: positive_class=ind
        #    else: negative_class=ind
        positive_class=np.where(self.classes_ == 1)[0][0]
        negative_class=np.where(self.classes_ == 0)[0][0]

        #this was not in the original implementation, should have an adaptive role
        binary_predictions=self.predict(X)
        self.group_weights=[len(X[binary_predictions==i]) / float(len(X)) for i in self.classes_]
        self.group_log_weights=np.log(self.group_weights)

        #normalize by class weight

        ##TODO recalculate group_weights


        ll_norm=np.vstack(ll)+ np.vstack(self.group_log_weights)
        #return np.vstack(ll).argmax(axis=0)
        #return np.vstack(ll)[positive_class]
        #return np.exp((ll_norm-logsumexp(ll_norm,axis=0))[positive_class])
        return np.exp((ll_norm-logsumexp(ll_norm,axis=0))).transpose()




class TrainingDataCreator(object):
    def __init__(self,query,exclude_chr=[]):
        '''

        :param query:     type Data from DataLoader
        :return:
        '''
        self._query=query
        self._input_feat=Config.D["INPUT_FEATURE"]
        self._output_feat=Config.D["OUTPUT_FEATURE"]
        self._exclude_chr=exclude_chr

    @staticmethod
    def _run_outliers(x):
         x['outliers'] = MinCovDet().fit(x[['m_raw',"a_raw"]]).support_
         return x

    def create(self,type="all",outliers=True,jobid=None,mode="original"):
        '''
        default behaviour is to create
        :return:
        '''

        input_feat=self._input_feat
        output_feat=self._output_feat
        exclude_chr=self._exclude_chr


        levels_to_stack=[i for i in self._query.df.columns.names if i!="feature"]

        t=self._query.df.stack(levels_to_stack)


        t = t.replace([np.inf, -np.inf], np.nan)

        ###t.dropna(inplace=True)
        #this assures chr is part of the input feat
        t.reset_index(level="Chr",inplace=True)
        t['Chr']=t['Chr'].astype('category')

        #t['gtype']=t['gtype'].astype('category')
        t=t[(t['gtype']!="NC") & (~t.Chr.isin(exclude_chr))]
        if type!="all":
            if type=="hom":
              t=t[(t['gtype']=="AA") | (t['gtype']=="BB")]
            elif type=="het":
               t=t[t['gtype']=="AB"]

        logging.info('Training Dataset properties' + str(Counter(t['gtype'])))
        if jobid:
          protocol.add_stat(jobid,str(Counter(t['gtype'])))
        else:  protocol.send_data('Dataset properties' + str(Counter(t['gtype'])))


        t=t[input_feat+[output_feat]].dropna()

        t[output_feat]=t[output_feat].astype(int)
        if Config.D["OUTLIERS"]:
          t=t.groupby(["Chr",'gtype']).apply(self._run_outliers)

        if 'gtype' in input_feat:
            #t['gtype']=t['gtype'].astype('category')
            t['gtype'].replace(Settings.Settings.CODE, inplace=True)
            t['gtype']=t['gtype'].astype('category')




        X=pd.get_dummies(t[input_feat]).values
        if Config.D["OUTLIERS"]:
          #Y=pd.get_dummies(((t.output.apply(int) + t.outliers.apply(int)) > 1).apply(int))
          Y=((t.output.apply(int) + t.outliers.apply(int)) > 1).apply(int)
        else:
          Y=t[output_feat].apply(int)
        #X=t[input_feat].values
        #Y=t[output_feat].values


        ####
        #here we use downsampling if needed
        ####
        nsamples=len(X)
        noutputs=len(Y)
        if nsamples != noutputs:
            raise ValueError('#samples != #outputs')




        #indices for all samples
        indall=range(0,nsamples)

        #number of positive samples
        npositive=sum(Y.values)

        #number of negative samples
        nnegative=len(Y.values)-npositive

        #if npositive<nnegative:
        #    raise ValueError('#positive<#negative in the input dataset!')



        cur_ratio=npositive/(nnegative*1.0)


        ratio=Config.D["RATIO"]

        if ratio==0:#keep the original dataset
           final_ind=indall
        else:
           #indices for positive samples
           indpositive=np.argwhere(Y.values==1)[:,0]
           #indices for negative samples
           indnegative=np.argwhere(Y.values==0)[:,0]

           nsamplepos=np.floor(npositive/(cur_ratio/(ratio*1.0)))
           #else: nsamplepos=np.floor(npositive*(cur_ratio/(ratio*1.0)))
           nsampleneg=nnegative

           selindpos=np.random.choice(indpositive,int(nsamplepos))
           selindneg=indnegative

           final_ind=np.concatenate((selindpos,selindneg))

        positive=sum(Y[final_ind])
        negative=len(Y[final_ind])-positive

        logging.info("Using training ratio {}".format(positive/(negative*1.0)))
        logging.info("Dataset: " + Config.D["ID"] +" positive:" + str(positive) + ", negative: " + str(negative))

        ###############
        return (X[final_ind],Y.values[final_ind])

    #method creates training data for
    def create_components(self,type="all",outliers=True,jobid=None):
        '''
        default behaviour is to create
        :return:
        '''

        input_feat=self._input_feat
        output_feat=self._output_feat
        exclude_chr=self._exclude_chr
        levels_to_stack=[i for i in self._query.df.columns.names if i!="feature"]
        t=self._query.df.stack(levels_to_stack)


        t = t.replace([np.inf, -np.inf], np.nan)

        ###t.dropna(inplace=True)
        #this assures chr is part of the input feat
        t.reset_index(level="Chr",inplace=True)
        t['Chr']=t['Chr'].astype('category')

        #t['gtype']=t['gtype'].astype('category')
        t=t[(t['gtype']!="NC") & (~t.Chr.isin(exclude_chr))]
        if type!="all":
            if type=="hom":
              t=t[(t['gtype']=="AA") | (t['gtype']=="BB")]
            elif type=="het":
               t=t[t['gtype']=="AB"]

        #tu by mala byt statistika!!!

        X=pd.get_dummies(t[input_feat].dropna()).values

        logging.info('Dataset properties' + str(Counter(t['gtype'])))
        if jobid:
          protocol.add_stat(jobid,str(Counter(t['gtype'])))
        else:  protocol.send_data('Dataset properties' + str(Counter(t['gtype'])))


        #covariancemat=t.groupby(["gtype","output"])["m","a"].cov().apply(lambda x: pd.DataFrame(np.linalg.pinv(x.values), x.columns, x.index) )
        #covariancemat=t.groupby(["gtype","output"])["m","a"].agg(lambda x:pd.DataFrame(np.linalg.pinv(x.cov()),x.cov().columns, x.cov().index ))



        #covariancemat=t.groupby(["gtype","output"])["m","a"].cov()
        #covariancemat=t.groupby(["gtype","output"])[Config.D["INPUT_FEATURE"]].cov()

        #means=t.groupby(["gtype","output"])[Config.D["INPUT_FEATURE"]].mean()

        #covariancemat.index=pd.MultiIndex.from_tuples([("_".join([i[0],str(i[1])]),i[2]) for i in covariancemat.index.values],names=["component","feature"])
        #covariancemat=covariancemat.swaplevel()



        covariance_ar=[]
        weights_ar=[]
        groups=[]
        means=[]
        #for i in range(covariancemat.to_panel().values.shape[1]):
        #    print np.linalg.pinv(covariancemat.to_panel().values[:,i])
        #    test_ar.append(np.linalg.pinv(covariancemat.to_panel().values[:,i]))


        for g,df in t.groupby(["gtype","output"]):
            covariance_ar.append(np.linalg.pinv((df[Config.D["INPUT_FEATURE"]].cov()).values))
            weights_ar.append(len(df))
            groups.append(g)
            means.append(df[Config.D["INPUT_FEATURE"]].mean())
        #return tuple of: (means,precision_matrix,weights)


        ###Adjusted version
        '''
        for g,df in t.groupby("output"):
            if g:
               for g2,df2 in df.groupby("gtype"):
                 covariance_ar.append(np.linalg.pinv((df2[Config.D["INPUT_FEATURE"]].cov()).values))
                 means.append(df2[Config.D["INPUT_FEATURE"]].mean())
                 weights_ar.append(len(df2))
                 groups.append(g2)
            else:
                covariance_ar.append(np.linalg.pinv((df[Config.D["INPUT_FEATURE"]].cov()).values))
                means.append(df[Config.D["INPUT_FEATURE"]].mean())
                weights_ar.append(len(df))
                groups.append('Null')
        '''
        ####################


        #return (X,means.values,np.array(covariance_ar),np.array([float(i)/sum(weights_ar) for i in weights_ar]),groups)
        return (X,np.array(means),np.array(covariance_ar),np.array([float(i)/sum(weights_ar) for i in weights_ar]),groups)


    def add_outliers(self,func=MinCovDet):
        pass
        #self._query.detect_outliers(func,self._input_feat)


class Trainer(object):
    def __init__(self,data,jobid=None):

      self._samples=data.get_samples_names()
      logging.info('Samples in the bucket:' + str(self._samples))
      if jobid:
          protocol.add_stat(jobid,sorted(self._samples))

      if Config.D["CLFNAME"]=="gaussian":
         #self._X,self._Y=TrainingDataCreator(data).create(outliers=Config.D["OUTLIERS"],type="het")
         #self._clf=mixture.GaussianMixture(n_components=2, covariance_type='full',means_init=[[np.mean(self._X[self._Y==0])] ,[np.mean(self._X[self._Y==1])]])
         self._X,self._Means,self._Y,weights,self._components=TrainingDataCreator(data).create_components(outliers=Config.D["OUTLIERS"],type="all",jobid=jobid)
         self._clf=mixture.GaussianMixture(n_components=len(weights), covariance_type='full',means_init=self._Means ,precisions_init=self._Y,weights_init=weights,verbose=0)

         #self._clf=mixture.GaussianMixture(n_components=6, covariance_type='full',verbose=2)

         #self._clf=mixture.GaussianMixture(n_components=6, covariance_type='full',means_init=self._Means,precisions_init=self._Y)
         #self._clf=mixture.GaussianMixture(n_components=5, covariance_type='full',means_init=self._Means,precisions_init=self._Y)
      else:
         #self._X,self._Y=TrainingDataCreator(data).create(outliers=Config.D["OUTLIERS"],type=Config.D["TRAINING_SELECTION"],jobid=jobid)
         self._X,self._Y=TrainingDataCreator(data).create(outliers=Config.D["OUTLIERS"],type="all",jobid=jobid)
       #self._clf = mixture.GaussianMixture(n_components=5, covariance_type='full')
      if Config.D["CLFNAME"]=='rf':
         #self._clf=RandomForestClassifier(n_jobs=-1,n_estimators=128,max_features=None)
         self._clf=RandomForestClassifier(n_jobs=-1,n_estimators=Config.D["NTREES"],max_features=None)
      elif Config.D["CLFNAME"]=='dt':
         self._clf=tree.DecisionTreeClassifier()
      elif Config.D["CLFNAME"]=='kde':
           self._clf = (KernelDensity(kernel='gaussian'),KernelDensity(kernel='gaussian'))
      elif Config.D["CLFNAME"]=="lr":
        self._clf= linear_model.LogisticRegression(n_jobs=-1)
      elif Config.D["CLFNAME"]=="knn":
         self._clf=KNeighborsClassifier(n_jobs=-1)
      elif Config.D["CLFNAME"]=="bgmm":
         self._clf=mixture.BayesianGaussianMixture(n_components=7,weight_concentration_prior=0.01)
      elif Config.D["CLFNAME"]=="mlp":
         self._clf= MLPClassifier()
      elif Config.D["CLFNAME"]=="vbgmmcl" or "vbgmmcl" in Config.D["CLFNAME"]:
          self._clf=GMMcl(n_components=3,core="vbgmm")
      elif Config.D["CLFNAME"]=="gmmcl" or "gmmcl" in Config.D["CLFNAME"] or "gda" in Config.D["CLFNAME"]:
          self._clf=GMMcl(n_components=3,core="gmm")
      logging.info("Initialised mode: " + Config.D["CLFNAME"])

       #self._clf=SVC()
       #self._clf=KNeighborsClassifier(n_jobs=-1)


       #self._names=["BB_False","AA_True","BB_True","AB_True","AA_False",]
       #self._names=["BB_True","BB_False","AB_True","AA_False","AA_True"]

    '''this method is for GaussianMixture

    def __train(self):
       self._clf.fit(self._X)
       t=sorted([i for i in enumerate(self._clf.means_)], key=lambda x: x[1])
       order=sorted([(ord,i[0]) for ord,i in enumerate(t)],key=lambda x:[x[1]])
       # t is [(ord,val)]
       self._ordered_names=[self._names[o[0]] for o in order]
       return self._clf
    '''

    def train(self):
        if Config.D["CLFNAME"]=='kde':
          self._clf[0].fit(self._X[self._Y==1])
          self._clf[1].fit(self._X[self._Y==0])
        elif Config.D["CLFNAME"]=='gaussian':
          #self._clf.fit(self._X)
          self._clf.fit(self._Means)

        #elif Config.D["CLFNAME"]=="vbgmmcl":
        #  self._clf.fit(self._X,groupvar="pred")

        else:
          self._clf.fit(self._X,self._Y)



    def predict_self(self):
        return self._clf.predict(self._X)

    def predict_self_proba(self):
        return self._clf.predict_proba(self._X)

    def predict(self,X):
        if Config.D["CLFNAME"]=='kde':
            pos=self._clf[0].score_samples(X)
            neg=self._clf[1].score_samples(X)
            return np.array([ 1 if x>0.6 else 0 for x in pos/(pos+neg)])

        elif Config.D["CLFNAME"]=="gaussian":
          return np.array([int(self._components[i][1]) for i in self._clf.predict(X)])
        else:
          return self._clf.predict(X)



    #to make this universal, GMM should also do binary classification
    def predict_proba(self,X):
        if Config.D["CLFNAME"]=='kde':
            pos=self._clf[0].score_samples(X)
            neg=self._clf[1].score_samples(X)
            return np.array([ 1 if x>0.6 else 0 for x in pos/(pos+neg)])

        elif Config.D["CLFNAME"]=="gaussian":
          return np.array([int(self._components[i][1]) for i in self._clf.predict_proba(X)])

        elif Config.D["CLFNAME"]=="bgmm":
          pass

        else:
          positive_class=np.where(self._clf.classes_==1)
          #return self._clf.predict_proba(X)[positive_class]
          return (self._clf.predict_proba(X)[:,positive_class]).flatten()


    #this methods takes only 1
    def create_roc_data_one_layer(self,test):
        levels_to_stack=[i for i in test.df.columns.names if i!="feature"]
        stacked=test.df.stack(level=levels_to_stack).reset_index(level="Chr")
        #exclude NCs
        #stacked=stacked[stacked.gtype!="NC"][Config.D["INPUT_FEATURE"]]
        #include only AB
        if Config.D["SELECTION"]=="all":
           stacked=stacked[stacked.gtype!="NC"]#[Config.D["INPUT_FEATURE"]]
        else:
           stacked=stacked[stacked.gtype=="AB"]#[Config.D["INPUT_FEATURE"]]

          #stacked = stacked.replace([np.inf, -np.inf], np.nan)
          #stacked=stacked.replace(np.nan,-200.0)


          #if "gtype" in Config.D["INPUT_FEATURE"]:
        stacked['gtype'].replace(Settings.Settings.CODE, inplace=True)
        stacked['gtype']=stacked['gtype'].astype('category')
        #if 'Chr' in Config.D["INPUT_FEATURE"]:
        stacked['Chr']=stacked['Chr'].astype('category')


        if Config.D["CLFNAME"]!="gaussian":
          for ind,val in enumerate(self.clf.classes_):
            if val==1: positive_class=ind
          stacked["_".join(["proba", Config.D["CLFNAME"]])]=self.predict_proba(pd.get_dummies(stacked[Config.D["INPUT_FEATURE"]]).replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0))[:,positive_class]
        else:
          gmm_pred=self.clf.predict_proba(stacked[Config.D["INPUT_FEATURE"]].replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0))
          #positive_gmm=np.take(gmm_pred,np.argwhere(np.array(self._components)!="Null"),axis=1).sum(axis=1)
          positive_gmm=np.take(gmm_pred,np.argwhere(np.array(self._components)[:,1]),axis=1).sum(axis=1)
          stacked["_".join(["proba", Config.D["CLFNAME"]])]=positive_gmm
          #component_ind=np.argwhere(np.array(self._components)[:,1]==1)[np.array(stacked["gtype"])]

          #probabilities of the gtype from GMM
          #probs_gmm=np.take(gmm_pred,component_ind[:,0])


        return stacked



    def create_roc_data(self,test):
        levels_to_stack=[i for i in test.df.columns.names if i!="feature"]
        stacked=test.df.stack(level=levels_to_stack).reset_index(level="Chr")
        #exclude NCs
        #stacked=stacked[stacked.gtype!="NC"][Config.D["INPUT_FEATURE"]]
        #include only AB
        if Config.D["SELECTION"]=="all":
           stacked=stacked[stacked.gtype!="NC"]#[Config.D["INPUT_FEATURE"]]
        else:
           stacked=stacked[stacked.gtype=="AB"]#[Config.D["INPUT_FEATURE"]]

          #stacked = stacked.replace([np.inf, -np.inf], np.nan)
          #stacked=stacked.replace(np.nan,-200.0)


          #if "gtype" in Config.D["INPUT_FEATURE"]:
        stacked['gtype'].replace(Settings.Settings.CODE, inplace=True)
        stacked['gtype']=stacked['gtype'].astype('category')
        #if 'Chr' in Config.D["INPUT_FEATURE"]:
        stacked['Chr']=stacked['Chr'].astype('category')

        for ind,val in enumerate(self.clf.classes_):
            if val==1: positive_class=ind

        lookup="_".join(["proba", Config.D["CLFNAME"]])


        X=Imputer().fit_transform(pd.get_dummies(stacked[Config.D["INPUT_FEATURE"]].replace([np.inf, -np.inf], np.nan)))


        #stacked[lookup]=self.predict_proba(pd.get_dummies(stacked[Config.D["INPUT_FEATURE"]]).replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0))[:,positive_class]
        stacked[lookup]=self.predict_proba(X)[:,positive_class]



        #we create GMM from values that are for sure true according to the classifier - probability 1.0
        # ... or for sure false - probability 0.0
        ##(components,gmm)=create_components_for_gmm_roc(stacked.query("proba_rf==1 or proba_rf==0")[["proba_rf","gtype","output"] + Config.D["INPUT_FEATURE"]])
        (components,gmm)=create_components_for_gmm_roc(stacked[[lookup,"gtype","output"] + Config.D["INPUT_FEATURE"]],threshold=0.5)

        #in the previous approach we only used means, lets try it again
        ##gmm.fit(stacked[Config.D["INPUT_FEATURE"]].replace([np.inf, -np.inf], np.nan).dropna())

        #gmm.fit(X)
        gmm.fit(gmm.means_init)

        #gmm.fit(stacked[Config.D["INPUT_FEATURE"]].replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0))


        #stacked["pred_gmm"]=gmm.predict_proba(stacked[Config.D["INPUT_FEATURE"]].replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0))
        gmm_pred=gmm.predict_proba(X)

        gmm_bin_pred=gmm.predict(X)

        #np.savetxt("foo.csv", gmm_pred, delimiter=",")
        #print "saved posterior prob..."
        #get indices of the components referring to positive clusters
        #component_ind=np.argwhere(np.array(components)[:,1]==1)[np.array(stacked["gtype"])]

        #probabilities of the gtype from GMM
        #probs_gmm=np.take(gmm_pred,component_ind[:,0])



        #stacked=stacked.assign(pred_gmm_component=np.array(components)[np.argmax(gmm_pred,axis=1)],pred_gmm=np.max(gmm_pred,axis=1))
        stacked=stacked.assign(pred_gmm_component=np.array(components)[np.argmax(gmm_pred,axis=1)][:,0],pred_gmm=np.max(gmm_pred,axis=1))
        #positive_gmm=np.take(gmm_pred,np.argwhere(np.array(components)!="Null"),axis=1).sum(axis=1)
        #positive_gmm=np.take(gmm_pred,np.argwhere(np.array(components)!='Null'),axis=1).sum(axis=1)
        positive_gmm=np.take(gmm_pred,np.argwhere(np.array(components)[:,1]==1),axis=1).sum(axis=1)


        #this is the yscore
        #stacked["proba_gmm"]=probs_gmm
        stacked["gmm_bin"]=gmm_bin_pred
        #stacked["proba_gmm_rf"]=probs_gmm*stacked["proba_rf"].values
        stacked["proba_gmm_"+ Config.D["CLFNAME"]]=np.maximum(positive_gmm[:,0],stacked[lookup].values)
        stacked["proba_positive_gmm_" + Config.D["CLFNAME"]]=positive_gmm[:,0]*stacked[lookup]
        stacked["positive_gmm"]=positive_gmm


        #this is the correct output

        #gmm.predict_proba(stacked[Config.D["INPUT_FEATURE"]].replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0))



        #return stacked[["gtype","output","gmm_bin","proba_" + Config.D["CLFNAME"],"proba_gmm","positive_gmm","proba_gmm_"+ Config.D["CLFNAME"],"proba_positive_gmm_"+ Config.D["CLFNAME"],"score"] + Config.D["INPUT_FEATURE"]]
        return stacked

    def predict_decorate(self,test,**kwargs):
        #stacked=test.df.stack("individual").reset_index(level="Chr")
        #this is more generic version:00001 = {float64} 0.869273134945
        levels_to_stack=[i for i in test.df.columns.names if i!="feature"]
        stacked=test.df.stack(level=levels_to_stack).reset_index(level="Chr")
        #exclude NCs
        #stacked=stacked[stacked.gtype!="NC"][Config.D["INPUT_FEATURE"]]
        #include only AB
        if Config.D["SELECTION"]=="all":
           stacked=stacked[stacked.gtype!="NC"]#[Config.D["INPUT_FEATURE"]]
        else:
           stacked=stacked[stacked.gtype=="AB"]#[Config.D["INPUT_FEATURE"]]

          #stacked = stacked.replace([np.inf, -np.inf], np.nan)
          #stacked=stacked.replace(np.nan,-200.0)


          #if "gtype" in Config.D["INPUT_FEATURE"]:
        stacked['gtype'].replace(Settings.Settings.CODE, inplace=True)
        stacked['gtype']=stacked['gtype'].astype('category')
        #if 'Chr' in Config.D["INPUT_FEATURE"]:
        stacked['Chr']=stacked['Chr'].astype('category')

        rec_name="{0}_ratio:{1}_pred".format(Config.D["CLFNAME"],Config.D["RATIO"])
        rec_name_prob="{0}_ratio:{1}_prob".format(Config.D["CLFNAME"],Config.D["RATIO"])

        stacked[rec_name]=self.predict(pd.get_dummies(stacked[Config.D["INPUT_FEATURE"]]).replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0))
        stacked[rec_name_prob]=self.predict_proba(pd.get_dummies(stacked[Config.D["INPUT_FEATURE"]]).replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0))

        stacked[rec_name_prob]=stacked[rec_name_prob].astype(float)


        if "threshold" in kwargs.keys():
            stacked[rec_name]=[1 if i>kwargs["threshold"] else 0 for i in stacked[rec_name_prob]]
        #else:
        #    stacked["pred"]=self.predict(pd.get_dummies(stacked[Config.D["INPUT_FEATURE"]]).replace([np.inf, -np.inf], np.nan).replace(np.nan,-200.0))


        #####load mixture model
        '''
        if Config.D["CLFNAME"]!="gaussian":
            gmm=mixture.GaussianMixture(n_components=2, covariance_type='full',means_init=[[np.mean(stacked[stacked.pred==0].a)] ,[np.mean(stacked[stacked.pred==1].a)]])
            gmm.fit(stacked[["a"]])
            stacked["pred_gauss"]=gmm.predict(stacked[["a"]])
        '''
        #Y_pred_gaus=gmm.predict([X_test[:,[1]]])


        #if 'Chr' in Config.D["INPUT_FEATURE"]:
        #stacked=stacked.set_index("Chr", drop=True, append=True).unstack(level=levels_to_stack).swaplevel(1,0,axis=1).swaplevel(2,1,axis=0)
        stacked=stacked.set_index("Chr", drop=True, append=True).unstack(level=levels_to_stack).reorder_levels(test.df.columns.names,axis=1).swaplevel(2,1,axis=0)



        stacked.sortlevel(axis=0,inplace=True,sort_remaining=True)
        stacked.sortlevel(axis=1,inplace=True,sort_remaining=True)

        '''
        if Config.D["CLFNAME"]!="gaussian":
          #o=pd.concat([test.df,stacked.loc[:,(slice(None),["pred","pred_gauss"])]],axis=1,join="inner")
          o=pd.concat([test.df,stacked.loc[:,(slice(None),["pred","pred_gauss"])]],axis=1)
        else:
          #o=pd.concat([test.df,stacked.loc[:,(slice(None),["pred"])]],axis=1,join="inner")
        '''
        if rec_name in test.df.columns.get_level_values(level="feature") or rec_name_prob in test.df.columns.get_level_values(level="feature"):
           #test.df.update(stacked.loc[:,(slice(None),["pred"])])
           #more generic version
           test.df.update(stacked.xs(rec_name,axis=1,drop_level=False,level="feature"))
           test.df.update(stacked.xs(rec_name_prob,axis=1,drop_level=False,level="feature"))
           #test.df.join(stacked.xs("pred",axis=1,drop_level=False,level="feature"))
        else:
            #o=pd.concat([test.df,stacked.loc[:,(slice(None),["pred"])]],axis=1)
            #more genetic version
            test.df=test.df.join(stacked.xs(rec_name,axis=1,drop_level=False,level="feature"))
            test.df=test.df.join(stacked.xs(rec_name_prob,axis=1,drop_level=False,level="feature"))
            test.df.sortlevel(axis=0,inplace=True,sort_remaining=True)
            test.df.sortlevel(axis=1,inplace=True,sort_remaining=True)
            #update
            #test.df=o
        #return Dataobject with decorated dataframe

        test.container.append(rec_name)
        test.container.append(rec_name_prob)
        return test



    @property
    def clf(self):
        return self._clf

    '''this method is again for GaussianMixture
    def __evaluate(self):
        components=self.predict_self()
        converted=[self._ordered_names[i] for i in components]
        return (Counter((converted==self._Y.T)[0])[True],Counter((converted==self._Y.T)[0])[False])
    '''

    def evaluate(self):
        pass

    def save_model(self):
        '''serialize model to file
        :return:
        '''
        with open("RF.model", "w") as f:
           print "serializing ML model..."
           pickle.dump(self._clf, f)


class SerializedTrainer(Trainer):
    def __init__(self,s_clf):
        self._clf=s_clf


class Tester(Trainer):
    def __init__(self,clfpath):
        with open(clfpath,"rb") as f:
           print "deserializing ML model..."
           self._clf = pickle.load(f)




class Validator(object):
    def __init__(self,model,data):
        '''
        Load model and testing data
        :param model:
        :param data:
        :return:
        '''
        pass


    def validate(self):
        pass



def save_state():
    gdna=Data.create_from_file(sys.argv[1],"GDNA",exclude=Settings.Settings.TO_REMOVE)
    sc=Data.create_from_file(sys.argv[2],"SC")

    sc.custom_filter("sc")
    mother=gdna.slice("gm07224")
    father=gdna.slice("gm07225")
    ref_proband=gdna.slice("gdna")
    p=Patterns.check_parental_trios(mother,father,ref_proband)
    ref_proband.add_lq_indices(list(p[p==False].index.values))
    sc.calculate_transformations()
    sc.compare_against_reference(ref_proband)
    sc.create_transition_matrices(ref_proband)

    #sc.compare_against_reference(ref_proband)


    sc.calculate_group_columns_index()


    with open("SC.obj", "w") as f:
        print "serializing SC..."
        pickle.dump(sc, f)


    with open("REF.obj", "w") as f:
        print "serializing reference gdna..."
        pickle.dump(ref_proband, f)

    print "Done..."



def deserialize_and_run():
    with open(Config.D["SCOBJECT"], 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
      ref_proband = pickle.load(f)
      print "finished loading object"
      print ref_proband.get_individuals_count()


def starting_procedure_load_sc():
    with open(Config.D["SCOBJECT"], "rb") as input_file:
      sc = cPickle.load(input_file)

    with open(Config.D["SCTRAINOBJECT"], "rb") as input_file:
      sc_train = cPickle.load(input_file)

    return (sc_train,sc)



'''
this should be generic method that returns the single cell Data object and gDNA Data object
'''
def starting_procedure():
    gdna=Data.create_from_file(Config.D["GDNA"],"GDNA",exclude=Settings.Settings.TO_REMOVE)
    logging.info("Loaded " + Config.D["GDNA"])
    gdna.apply_NC_threshold_3(0.15,inplace=True)
    logging.info("Applied  threshold to gDNA")

    sc=Data.create_from_file(Config.D["SC"],"SC")
    logging.info("Loaded " + Config.D["SC"])

    for d in [gdna,sc]: d.restrict_chromosomes(Config.D["CHROMOSOMES"])
    logging.info("Restricted to chromosomes defined in {} ...".format(Config.D["CHROMOSOMES"]))


    #gdna.df=gdna.df.loc[(slice(None),Config.D["CHROMOSOMES"],slice(None)),:]
    #sc.df=sc.df.loc[(slice(None),Config.D["CHROMOSOMES"],slice(None)),:]
    #sc.df=sc.df.loc[(slice(None),Config.D["CHROMOSOMES"],slice(None)),:]
    #gdna.df=gdna.df.loc[(slice(None),Config.D["CHROMOSOMES"],slice(None)),:]
    sc.custom_filter("sc")
    mother=gdna.slice("gm07224")
    father=gdna.slice("gm07225")
    ref_proband=gdna.slice("gdna")
    p=Patterns.check_parental_trios(mother,father,ref_proband)
    logging.info("Parental patterns checked...")
    ref_proband.add_lq_indices(list(p[p==False].index.values))


    #protocol.send_data("Reference gDNA parameters:\n" + ref_proband.get_call_rates_consensus().to_string())
    protocol.send_data("Reference gDNA parameters:\n" + ref_proband.get_call_rates_consensus().transform(lambda x: x/sum(x)).to_string())

    #first=min(Config.D["SCORE_THRESHOLD"],Config.D["TRAINING_SCORE_THRESHOLD"])
    #second=max(Config.D["SCORE_THRESHOLD"],Config.D["TRAINING_SCORE_THRESHOLD"])


    sc.calculate_transformations_2()
    sc.compare_against_reference(ref_proband)



          #return (sc_transition_matrix,gdna_transition_matrix)


    '''
    if Config.D["SCORE_THRESHOLD"]>=Config.D["TRAINING_SCORE_THRESHOLD"]:#training score is less restrictive, so it will keep more values
        sc_train=sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=False)
        sc.apply_NC_threshold_3(Config.D["SCORE_THRESHOLD"],inplace=True)
        second=Config.D["SCORE_THRESHOLD"]
    else:
        sc.apply_NC_threshold_3(Config.D["SCORE_THRESHOLD"],inplace=True)
        sc_train=sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=False)

    sc.compare_against_reference(ref_proband)
    sc_train.compare_against_reference(ref_proband)

    #with open(Config.D["SCOBJECT"], "wb") as output_file:
    #  cPickle.dump(sc, output_file)

    #with open(Config.D["SCTRAINOBJECT"], "wb") as output_file:
    #  cPickle.dump(sc_train, output_file)
    '''
    #return (sc_train,sc,ref_proband)
    return (sc,ref_proband)


def starting_procedure_T21():
    gdna=Data.create_from_file(Config.D["GDNA"],"GDNA",exclude=Settings.Settings.TO_REMOVE + ["GenTrain Score"])
    logging.info("Loaded " + Config.D["GDNA"])
    gdna.apply_NC_threshold_3(0.15,inplace=True)
    logging.info("Applied  threshold to gDNA")
    #gdna.consensus_genotype()


    sc=Data.create_from_file(Config.D["SC"], "SC", exclude=["GenTrain Score"])
    ####THIS IS USUALLY 0.01
    sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=True)

    logging.info("Loaded " + Config.D["SC"])
    for d in [gdna,sc]: d.restrict_chromosomes(Config.D["CHROMOSOMES"])
    logging.info("Restricted to autosomes...")

    ref_proband=gdna

    #protocol.send_data("Reference gDNA parameters:\n" + ref_proband.get_call_rates_consensus().to_string())
    #protocol.send_data("Reference gDNA parameters:\n" + ref_proband.get_call_rates_consensus().transform(lambda x: x/sum(x)).to_string())

    #first=min(Config.D["SCORE_THRESHOLD"],Config.D["TRAINING_SCORE_THRESHOLD"])
    #second=max(Config.D["SCORE_THRESHOLD"],Config.D["TRAINING_SCORE_THRESHOLD"])


    sc.calculate_transformations_2()
    sc.compare_against_reference(ref_proband)

    #return (sc_transition_matrix,gdna_transition_matrix)


    '''
    if Config.D["SCORE_THRESHOLD"]>=Config.D["TRAINING_SCORE_THRESHOLD"]:#training score is less restrictive, so it will keep more values
        sc_train=sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=False)
        sc.apply_NC_threshold_3(Config.D["SCORE_THRESHOLD"],inplace=True)
        second=Config.D["SCORE_THRESHOLD"]
    else:
        sc.apply_NC_threshold_3(Config.D["SCORE_THRESHOLD"],inplace=True)
        sc_train=sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=False)

    sc.compare_against_reference(ref_proband)
    sc_train.compare_against_reference(ref_proband)

    #with open(Config.D["SCOBJECT"], "wb") as output_file:
    #  cPickle.dump(sc, output_file)

    #with open(Config.D["SCTRAINOBJECT"], "wb") as output_file:
    #  cPickle.dump(sc_train, output_file)
    '''
    #return (sc_train,sc,ref_proband)
    return (sc,ref_proband)


def starting_procedure_GM12878():
    sc=Data.create_from_file(Config.D["SC"], "SC", exclude=["GenTrain Score"])
    logging.info("Loaded " + Config.D["SC"])
    gdna=sc.slice("gdna")
    logging.info("Loaded gDNA by slicing..." )

    sc=sc.remove("gdna")
    logging.info("removed gDNA from SC dataset" )
    ####THIS IS USUALLY 0.01
    sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=True)
    gdna.apply_NC_threshold_3(0.15,inplace=True)
    logging.info("Applied  threshold to gDNA")





    for d in [gdna,sc]: d.restrict_chromosomes(Config.D["CHROMOSOMES"])
    logging.info("Restricted to autosomes...")

    ref_proband=gdna

    #protocol.send_data("Reference gDNA parameters:\n" + ref_proband.get_call_rates_consensus().to_string())
    ######protocol.send_data("Reference gDNA parameters:\n" + ref_proband.get_call_rates_consensus().transform(lambda x: x/sum(x)).to_string())

    #first=min(Config.D["SCORE_THRESHOLD"],Config.D["TRAINING_SCORE_THRESHOLD"])
    #second=max(Config.D["SCORE_THRESHOLD"],Config.D["TRAINING_SCORE_THRESHOLD"])


    sc.calculate_transformations_2()
    sc.compare_against_reference(ref_proband)

    #return (sc_transition_matrix,gdna_transition_matrix)


    '''
    if Config.D["SCORE_THRESHOLD"]>=Config.D["TRAINING_SCORE_THRESHOLD"]:#training score is less restrictive, so it will keep more values
        sc_train=sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=False)
        sc.apply_NC_threshold_3(Config.D["SCORE_THRESHOLD"],inplace=True)
        second=Config.D["SCORE_THRESHOLD"]
    else:
        sc.apply_NC_threshold_3(Config.D["SCORE_THRESHOLD"],inplace=True)
        sc_train=sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"],inplace=False)

    sc.compare_against_reference(ref_proband)
    sc_train.compare_against_reference(ref_proband)

    #with open(Config.D["SCOBJECT"], "wb") as output_file:
    #  cPickle.dump(sc, output_file)

    #with open(Config.D["SCTRAINOBJECT"], "wb") as output_file:
    #  cPickle.dump(sc_train, output_file)
    '''
    #return (sc_train,sc,ref_proband)
    return (sc,ref_proband)





def score_range(start, end, step):
    while start <= end:
        yield start
        start += step


def tm_routine(container,ref,sc,score,run,type):
  sc_transition_matrix,gdna_transition_matrix,raw_transition_matrix=sc.create_transition_matrices(ref)
  container[(score,run,type,"sc")]=sc_transition_matrix
  container[(score,run,type,"gdna")]=gdna_transition_matrix
  container[(score,run,type,"transition_count")]=raw_transition_matrix


def get_optimal_threshold(sbj):
  d={}
  for g,df in sbj.groupby(["type","curve"]):
    optimal_idx=np.argmax(df["tpr_recall"]-df["fpr_precision"])
    d[g]=df["threshold"][optimal_idx]
  return d



'''

'''
#def evaluate_metrics(df,func=[metrics.precision_score,metrics.recall_score,metrics.accuracy_score]):
def evaluate_metrics(df,colname,thr,names=["precision"],func=[metrics.precision_score]):
    #stack and remove na
    #colname="{}_proba".format(Config.D["CLFNAME"])
    stacked=df.stack(level=0)[["output",colname]].dropna()

    for n,f in zip(names,func):
      if n not in ["roc_auc_score"]:
        yield n,f(stacked["output"].apply(int),stacked[colname]>=thr)
      else:
        yield n,f(stacked["output"].apply(int),stacked[colname])







SAVE=True

if __name__ == "__main__":
  np.warnings.filterwarnings('ignore')
  #jb.Parallel._print = _print


  #load config
  Config.load(sys.argv[1])
  os.chdir(os.path.dirname(os.path.abspath(sys.argv[1])))

  #init logger
  logging.basicConfig(format='%(asctime)s %(message)s',filename=Config.D["LOGFILE"], level=logging.DEBUG)
  logging.info("Started using " + sys.argv[1])




  #sc_train,sc,ref=starting_procedure()
  #############
  #sc,ref=starting_procedure()
  if Config.D["ID"]=="T21":  sc,ref=starting_procedure_T21()
  elif Config.D["ID"]=="7228": sc,ref=starting_procedure()
  elif Config.D["ID"]=="GM12878": sc,ref=starting_procedure_GM12878()

  #sc_train,sc=starting_procedure_load_sc()
  #sc=starting_procedure_load_sc()



  outputvar=[]
  transition_matrix_container=dict()

  tm_routine(transition_matrix_container,ref,sc,Config.D["TRAINING_SCORE_THRESHOLD"],-1,"unfiltered")
  logging.info("Score: {}, Run nr.: {}".format(Config.D["TRAINING_SCORE_THRESHOLD"],-1))

  #Config.D["CROSSFOLD"]=True
  if "CROSSFOLD" in Config.D.keys() and Config.D["CROSSFOLD"]:
    sc.apply_NC_threshold_3(Config.D["TRAINING_SCORE_THRESHOLD"], inplace=True)
    ar=[]
    for runnr,train,test in sc.stratify(10,revert=False):
         training=Trainer(train,jobid=str(runnr))
         training.train()
         test = training.predict_decorate(test)
         for res in evaluate_metrics(test.df,
                            names=["precision","recall","f1score","accuracy"],
                            func=[metrics.precision_score,metrics.recall_score,metrics.f1_score,metrics.accuracy_score]):
           #print tuple([runnr] + list(res))
          ar.append(tuple([runnr,Config.D["CLFNAME"]] + list(res)))

          #now its time for GMM
         if Config.D["CLFNAME"] != "gaussian":
              test.apply_prediction_to_output()

              tmpclf = Config.D["CLFNAME"]
              tmpfeatures = Config.D["INPUT_FEATURE"]

              Config.D["CLFNAME"] = "gaussian"
              Config.D["INPUT_FEATURE"] = ["m", "a"]
              training_gaussian = Trainer(test, jobid=str(runnr) + "_gaussian")
              training_gaussian.train()
              test = training_gaussian.predict_decorate(test)
              ####this changes the call rate by adding extra NCs!!!!
              test.apply_prediction_results()
              Config.D["CLFNAME"] = tmpclf
              Config.D["INPUT_FEATURE"] = tmpfeatures

              for res in evaluate_metrics(test.df,
                            names=["precision","recall","f1score","accuracy"],
                            func=[metrics.precision_score,metrics.recall_score,metrics.f1_score,metrics.accuracy_score]):
                 ar.append(tuple([runnr,Config.D["CLFNAME"] + "_gmm"]  + list(res)))



    cf_df=pd.DataFrame.from_records(ar,columns=["run","algorithm","metrics","val"])
    cf_df.to_csv(Config.D["CROSSFOLDOUT"])




    #skf=StratifiedKFold(n_splits=2, random_state=None, shuffle=False)

    #skf.split()




  elif Config.D["VALIDATION"]:
    protocol.init(Config.D["TITLE"],Config.D["DATANAME"])

    datalist=copy.deepcopy(sc.df.columns.get_level_values(level="individual").unique().values)
    ###print "--------------------"
    '''
    if Config.D["CLFNAME"]!="gaussian":
      print "n\tAdi_before\tAdi_after\tAdi_after_gauss\tAdo_before\tAdo_after\tFilter kept\tFilter kept gauss\tPrecision"
    else:
     print "n\tAdi_before\tAdi_after\tAdo_before\tAdo_after\tFilter kept\tPrecision"
    '''
    for actual_score in score_range(Config.D["SCORE_START"],Config.D["SCORE_END"],step=Config.D["SCORE_STEP"]):
        #sc.apply_NC_threshold_3(actual_score)

        #NUMBER OF REPLICATES
        for i in xrange(0,Config.D["NRUNS"]):
          logging.info("Score: {}, Run nr.: {}".format(actual_score,i))

          random.shuffle(datalist)
          #problem with this is that
          #train,_=sc_train.split_up_par(datalist,n=0.7)
          #_,test=sc.split_up_par(datalist,n=0.7)
          train,test=sc.split_up_par(datalist,Config.D["SPLIT"])
          test_score_experiment=copy.deepcopy(test)

          protocol.add_stat(str(i),test.get_samples_names())
          ###############
          training=Trainer(train,jobid=str(i))
          training.train()

          test.apply_NC_threshold_3(actual_score, inplace=True)

          test = training.predict_decorate(test)

          #a=evaluate_metrics(test.df,func=metrics.precision_score)

          ###This saves the original output
          outputvar.append(test.aggregate_flag_variables(ref))
          outputvar[-1].name = (i, 0, actual_score)
          tm_routine(transition_matrix_container,ref,test,actual_score,i,"unfiltered")
          # outputvar[-1]['filtered']=0
          # outputvar[-1]['run']=i

          ####This saves the predicted output
          #### this changes the call rate by adding extra NCs!!!!
          # we do this to prevent from loosing calls
          test_gaussian = copy.deepcopy(test)

          test.apply_prediction_results()
          ##############################3
          test.compare_against_reference(ref)
          outputvar.append(test.aggregate_flag_variables(ref))
          outputvar[-1].name = (i, 1, actual_score)
          tm_routine(transition_matrix_container,ref,test,actual_score,i,"filtered")


          # outputvar[-1]['filtered']=1
          # outputvar[-1]['run']=i

          ###This is 2nd level that check whether we can improve the prediction by Gaussian - only applies if the 1st layer was not gaussian
          if Config.D["CLFNAME"] != "gaussian":
              test_gaussian.apply_prediction_to_output()

              tmpclf = Config.D["CLFNAME"]
              tmpfeatures = Config.D["INPUT_FEATURE"]

              Config.D["CLFNAME"] = "gaussian"
              Config.D["INPUT_FEATURE"] = ["m", "a"]
              training_gaussian = Trainer(test_gaussian, jobid=str(i) + "_gaussian")
              training_gaussian.train()
              test_gaussian = training_gaussian.predict_decorate(test_gaussian)
              ####this changes the call rate by adding extra NCs!!!!
              test_gaussian.apply_prediction_results()
              ###############################
              test_gaussian.compare_against_reference(ref)
              outputvar.append(test_gaussian.aggregate_flag_variables(ref))
              outputvar[-1].name = (str(i) + "_gaussian", 1, actual_score)
              tm_routine(transition_matrix_container,ref,test_gaussian,actual_score,i,"gaussian_filtered")


              Config.D["CLFNAME"] = tmpclf
              Config.D["INPUT_FEATURE"] = tmpfeatures


          '''
          #this part is for checking whether we can get better precision by increasing the score
          for j in xrange(2,10):
            score_threshold=j/float(10)
            test_score_experiment.apply_NC_threshold_3(score_threshold)
            test_score_experiment.compare_against_reference(ref)
            outputvar.append(test_score_experiment.aggregate_flag_variables(ref))
            outputvar[-1].name=(i,0,score_threshold)
          '''



          '''
          X_test,Y_test=TrainingDataCreator(test).create(type=Config.D["SELECTION"],outliers=False)
          Y_pred=training.predict(X_test)
          '''




          #test.aggregate_flag_variables_by_chr(ref=ref)
          #test.aggregate_flag_variables_by_sample(ref=ref)



          '''
          final_o=o.df.stack("individual")
          '''
          #final_o['gtype'].replace(Settings.Settings.DECODE, inplace=True)
          #final_o.to_csv("/data/OUTPUTS/" + str(i+1)+Config.D["RAWPREFIX"],sep=";")


          #final_o["pred"].dropna().apply(bool).values
          '''
          if Config.D["CLFNAME"]!="gaussian":
            final_o=final_o[["pred","pred_gauss","adi","ado"]].dropna()
          else:
            final_o=final_o[["pred","adi","ado"]].dropna()
          '''

        to_out=pd.DataFrame(outputvar)
        to_out.index=pd.MultiIndex.from_tuples(to_out.index, names=['run', 'filtered','score'])
        to_out.sort_index(inplace=True,axis=1)
        ###
        #save table
        ###

        to_out.to_csv(Config.D["TABLE"],sep=";")
        pd.concat(transition_matrix_container,axis=0).to_csv(Config.D["MATRIX"],sep=";")
        protocol.send_data(to_out.to_string())
        protocol.output()

  elif Config.D["TRAIN"]:#train classifier
        #store the raw data to binary object
        #sc.df.stack(level=0).to_msgpack(Config.D["GCBIN"])
        #if Config.D["TRAIN"]:
        training=Trainer(sc)
        training.train()

        if Config.D["CLFNAME"]=="bgmm" or Config.D["CLFNAME"]=="gaussian":
              training.predict_decorate(sc).df.stack(level=0).to_msgpack(Config.D["GCBIN"])
              #training.clf.bic


        #save classifier
        with open(Config.D["CLASSIFIER"], "wb") as output_file:
              #cPickle.dump(training, output_file)
              cPickle.dump(training.clf,output_file)
        print "Classifier succesfully trained... saving to file..."


  else:#load classifier (deserialize) and predict for current single cell data
     with open(Config.D["CLASSIFIER"], "rb") as input_file:
        clf = cPickle.load(input_file)
        ###print clf.clf.get_params()
        #sc.apply_NC_threshold_3(Config.D["SCORE_START"])

        if "ROC" in Config.D.keys():
          ###rocdata=clf.create_roc_data(sc)
          ###rocdata.to_msgpack(sys.argv[1].split(".")[0]+ ".rocdata.bin")
          rocdata=pd.read_msgpack(sys.argv[1].split(".")[0]+ ".rocdata.bin")

          voter=LogisticRegression(n_jobs=-1)
          voter2=LogisticRegression(n_jobs=-1)
          voter3=LogisticRegression(n_jobs=-1)

          voter.fit(rocdata[["proba_" + Config.D["CLFNAME"],"score","proba_positive_gmm_" + Config.D["CLFNAME"]]],rocdata["output"])
          voter2.fit(rocdata[["proba_" + Config.D["CLFNAME"],"positive_gmm"]],rocdata["output"])
          voter3.fit(rocdata[["proba_" + Config.D["CLFNAME"],"score","positive_gmm"]],rocdata["output"])
          rocdata["ensemble"]=voter.predict_proba(rocdata[["proba_" + Config.D["CLFNAME"],"score","proba_positive_gmm_" + Config.D["CLFNAME"]]])[:,[1]]
          rocdata["ensemble2"]=voter2.predict_proba(rocdata[["proba_" + Config.D["CLFNAME"],"positive_gmm"]])[:,[1]]
          rocdata["ensemble3"]=voter3.predict_proba(rocdata[["proba_" + Config.D["CLFNAME"],"score","positive_gmm"]])[:,[1]]

          #rocdata=clf.create_roc_data_one_layer(sc)

          rocdata.to_csv(Config.D["TABLE"])

          ###
          #"proba_" + Config.D["CLFNAME"],\ -> 1st layer clf
          #"proba_gmm",\                    -> gmm only
          #"proba_gmm_"+ Config.D["CLFNAME"],\ -> 1st layer clf + gbb
          #"score" ,\                           -> Gencall
          #"positive_gmm",\                     -> gmm - probability is sum of positive classes
          #"proba_positive_gmm_"+ Config.D["CLFNAME"] -> 1st layer and gmm as sum of positive classes
          ########

          stack=[]

          with open(sys.argv[1].split(".")[0]+ ".f1score.perf","w") as fperf:
            ##fperf.write(str(clf.clf.get_params()))
            fperf.write("\n")
            recs=[]
            for region in "het","homo","all":
              if region=="all":
                  rocdata_reg=rocdata
              elif region=="het":
                  rocdata_reg=rocdata[rocdata["gtype"]==Settings.Settings.CODE["AB"]]
              else:
                  rocdata_reg=rocdata[rocdata["gtype"]!=Settings.Settings.CODE["AB"]]

              for column in ["proba_" + Config.D["CLFNAME"],
                          #"proba_gmm",
                          ##"proba_gmm_"+ Config.D["CLFNAME"],
                          "score" ,
                          #"gmm_bin",
                          #"positive_gmm",
                          "proba_positive_gmm_"+ Config.D["CLFNAME"],
                          "ensemble",
                          "ensemble2",
                          "ensemble3"]:
                fpr,tpr,threshold=metrics.roc_curve(rocdata_reg["output"],rocdata_reg[column])
                df_roc=pd.DataFrame(np.stack((fpr,tpr,threshold),axis=1),columns=["fpr_precision","tpr_recall","threshold"])
                df_roc["type"]=column + "_".join(Config.D["INPUT_FEATURE"])
                df_roc["curve"]="roc"
                df_roc["region"]=region
                stack.append(df_roc)

                precision, recall, thresholds = precision_recall_curve(rocdata_reg["output"],rocdata_reg[column])
                thresholds=np.append(thresholds,1)
                df_pr=pd.DataFrame(np.stack((precision,recall,thresholds),axis=1),columns=["fpr_precision","tpr_recall","threshold"])

                df_pr["type"]=column + "_".join(Config.D["INPUT_FEATURE"])
                df_pr["curve"]="pr"
                df_pr["region"]=region
                df_pr["fbeta"]=(1.25*precision*recall)/(0.25*precision+recall)

                stack.append(df_pr)

                optimal_idx=np.argmax(tpr-fpr)
                #optimal_idx2=np.argmin(recall-precision)
                optimal_idx2=np.argmin(np.sqrt((1-recall)*(1-recall)+(1-precision)*(1-precision)))

                optimal_threshold=thresholds[optimal_idx]
                #optimal_threshold_pr=thresholds[optimal_idx2]
                optimal_threshold_pr= thresholds[np.argmax((1.25*precision*recall)/(0.25*precision+recall))]
                #optimal_threshold_harmonic=thresholds[np.argmax([f1_score(rocdata_reg["output"],rocdata_reg[column]>=i)  for i in thresholds])]




                #(func[1],column,region,"optimal",optimal_threshold,func[0](rocdata_reg["output"], rocdata_reg[column]>optimal_threshold))



                if column in ["score","proba_" + Config.D["CLFNAME"],"proba_positive_gmm_"+ Config.D["CLFNAME"],"ensemble","ensemble2","ensemble3"]:

                  if column=="score":
                    threshold=0.15
                  else:
                    threshold=0.5



                  print column
                  #fperf.write(column + "\n")

                  for func in [#(classification_report,"clf report"),
                               (metrics.precision_score,"precision"),
                               (accuracy_score,"accuracy"),
                               (recall_score,"recall"),
                               (roc_auc_score,"roc_auc_score"),
                               (f1_score,"f1_score"),
                               (average_precision_score,"average_prec_score")]:

                    if func[1]=="roc_auc_score":
                      recs.append((func[1],column,region,"default",-1,func[0](rocdata_reg["output"], rocdata_reg[column])))
                    if func[1]=="average_prec_score":
                      recs.append((func[1],column,region,"default",-1,func[0](rocdata_reg["output"], rocdata_reg[column],average="weighted")))
                      #fperf.write(str(func[1])+":" +  str(func[0](rocdata_reg["output"], rocdata_reg[column])) + "\n")
                    else:
                      recs.append((func[1],column,region,"optimal",optimal_threshold,func[0](rocdata_reg["output"], rocdata_reg[column]>=optimal_threshold)))
                      recs.append((func[1],column,region,"default",threshold,func[0](rocdata_reg["output"], rocdata_reg[column]>=threshold)))
                      recs.append((func[1],column,region,"optimal_pr",optimal_threshold_pr,func[0](rocdata_reg["output"], rocdata_reg[column]>=optimal_threshold_pr)))
                      #recs.append((func[1],column,region,"optimal_harmonic",optimal_threshold_harmonic,func[0](rocdata_reg["output"], rocdata_reg[column]>=optimal_threshold_harmonic)))
                  fperf.write(column +  " default_threshold "  +  region + "\n")
                  fperf.write(str(confusion_matrix(rocdata_reg["output"], rocdata_reg[column]>=threshold)) + "\n")
                  fperf.write(column +  " optimal_threshold " + region   + "\n")
                  fperf.write(str(confusion_matrix(rocdata_reg["output"], rocdata_reg[column]>=optimal_threshold)) + "\n")
                  fperf.write(column +  " optimal_threshold_pr " + region   + "\n")
                  fperf.write(str(confusion_matrix(rocdata_reg["output"], rocdata_reg[column]>=optimal_threshold_pr)) + "\n")
                  #fperf.write(column +  " optimal_threshold_harmonic " + region   + "\n")
                  #fperf.write(str(confusion_matrix(rocdata_reg["output"], rocdata_reg[column]>=optimal_threshold_harmonic)) + "\n")


                      #fperf.write("Optimal threshold " + str(func[1])+":" +  str(func[0](rocdata_reg["output"], rocdata_reg[column]>optimal_threshold)) + "\n")
                      #fperf.write("Default threshold " + str(func[1])+":" +  str(func[0](rocdata_reg["output"], rocdata_reg[column]>threshold)) + "\n")
                  #fperf.write("Optimal threshold:" + str(optimal_threshold) + "\n")
                  #fperf.write("Default threshold:" + str(threshold) + "\n")
                  #fperf.write("--------------------------\n")
                  #print func[1]+":", func[0](rocdata["output"], rocdata[column]>threshold)


                #metrics.precision_score(rocdata.query('gtype==2')['output'],rocdata.query('gtype==2')["proba_rf"]>0.5)

              #data for the table:

          #"proba_positive_gmm_"+ Config.D["CLFNAME"]]


          pd.concat(stack).to_csv(Config.D["ROC"],mode="w")
          metrics_dump=pd.DataFrame.from_records(recs,columns=["metrics","classifier","region","threshold","threshold_value","value"])
          metrics_dump.to_csv(sys.argv[1].split(".")[0]+ ".f1score.perf.table",index=False)


          #optimal_threshold=get_optimal_threshold(pd.concat(stack))

          '''
          #metrics.roc_curve(rocdata[1].values,rocdata[0], pos_label=1)
          ###roc for rf
          fpr,tpr,threshold=metrics.roc_curve(rocdata["output"],rocdata["_".join(["proba",Config.D["CLFNAME"]])])
          #save to file
          rf_df=pd.DataFrame(np.stack((fpr,tpr,threshold),axis=1),columns=["fpr","tpr","threshold"])
          rf_df["type"]=Config.D["CLFNAME"]
          #rf_df["curve"]="roc"

          #precision, recall, thresholds = precision_recall_curve(rocdata["output"],rocdata["_".join(["proba",Config.D["CLFNAME"]])])

          #rf_df.to_csv(Config.D["ROC"],mode="w")

          #roc for gmm
          fpr,tpr,threshold=metrics.roc_curve(rocdata["output"],rocdata["proba_gmm"])
          gmm_df=pd.DataFrame(np.stack((fpr,tpr,threshold),axis=1),columns=["fpr","tpr","threshold"])
          gmm_df["type"]="gmm"

          ###roc for rf_gmm
          fpr,tpr,threshold=metrics.roc_curve(rocdata["output"],rocdata["proba_gmm_"+ Config.D["CLFNAME"]])
          rf_gmm_df=pd.DataFrame(np.stack((fpr,tpr,threshold),axis=1),columns=["fpr","tpr","threshold"])
          rf_gmm_df["type"]=Config.D["CLFNAME"] + "_gmm"

          ###roc for GenCall
          fpr,tpr,threshold=metrics.roc_curve(rocdata["output"],rocdata["score"])
          gencall_df=pd.DataFrame(np.stack((fpr,tpr,threshold),axis=1),columns=["fpr","tpr","threshold"])
          gencall_df["type"]="gencall"

          #roc for positive gmm
          #return stacked[["gtype","output","gmm_bin","proba_rf","proba_gmm","positive_gmm","proba_gmm_rf","proba_positive_gmm_rf","score"] + [Config.D["INPUT_FEATURE"]]]
          fpr,tpr,threshold=metrics.roc_curve(rocdata["output"],rocdata["positive_gmm"])
          positive_gmm=pd.DataFrame(np.stack((fpr,tpr,threshold),axis=1),columns=["fpr","tpr","threshold"])
          positive_gmm["type"]="positive_gmm"

          fpr,tpr,threshold=metrics.roc_curve(rocdata["output"],rocdata["proba_positive_gmm_"+ Config.D["CLFNAME"]])
          positive_gmm_rf=pd.DataFrame(np.stack((fpr,tpr,threshold),axis=1),columns=["fpr","tpr","threshold"])
          positive_gmm_rf["type"]="proba_positive_gmm_" + Config.D["CLFNAME"]

          pd.concat([rf_df,gmm_df,rf_gmm_df,gencall_df,positive_gmm,positive_gmm_rf]).to_csv(Config.D["ROC"],mode="w")
          #pd.concat([rf_df,gencall_df]).to_csv(Config.D["ROC"],mode="w")
          '''

        else:
            #report unfiltered properties
            tm_routine(transition_matrix_container,ref,sc,Config.D["SCORE_START"],-1,"unfiltered")

            if float(Config.D["SCORE_START"])!=0.15:
              #now apply standard 0.15 score to a backup
              sc_for_gencall=copy.deepcopy(sc)
              sc_for_gencall.apply_NC_threshold_3(inplace=True,sthreshold=0.15)
              #sc_for_gencall.compare_against_reference(inplace=True)
              tm_routine(transition_matrix_container,ref,sc_for_gencall,"0.15",-1,"gencall_filtered")
              ##########################
            else:
              sc.apply_NC_threshold_3(inplace=True,sthreshold=0.15)
              tm_routine(transition_matrix_container,ref,sc,"0.15",-1,"gencall_filtered")

            ###apply 1st layer



            sc = clf.predict_decorate(sc)

            #and report values for GMM estimation
            sc.apply_prediction_to_output()

            test_gaussian = copy.deepcopy(sc)

            sc.apply_prediction_results()
            #sc.compare_against_reference(ref)
            tm_routine(transition_matrix_container,ref,sc,Config.D["SCORE_START"],-1,"filtered")


            tmpclf = Config.D["CLFNAME"]
            tmpfeatures = Config.D["INPUT_FEATURE"]

            Config.D["CLFNAME"] = "gaussian"
            Config.D["INPUT_FEATURE"] = ["m", "a"]

            training_gaussian = Trainer(test_gaussian, jobid="_gaussian")
            training_gaussian.train()
            test_gaussian = training_gaussian.predict_decorate(test_gaussian)

            #rocdata=training_gaussian.create_roc_data(test_gaussian)

            test_gaussian.apply_prediction_results()
            #test_gaussian.compare_against_reference(ref)
            tm_routine(transition_matrix_container,ref,test_gaussian,Config.D["SCORE_START"],-1,"gaussian_filtered")


            Config.D["CLFNAME"] = tmpclf
            Config.D["INPUT_FEATURE"] = tmpfeatures



            pd.concat(transition_matrix_container,axis=0).to_csv(Config.D["MATRIX"],sep=";")




        #with open("SC.obj", "w") as f:
        #    print "serializing SC..."
        #    pickle.dump(sc, f)
