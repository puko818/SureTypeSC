'''DataLoader.py
Author: Ivan Vogel
Purpose: to encapsulate all Single Cell, Trios and Embryos Loading
'''

'''
Libraries to import
'''

import Config
import sys
import pandas as pd
import numpy as np
import re
import logging
import itertools

from collections import Counter
from math import floor

import Settings
from Settings import Parser
from  __builtin__ import any as b_any
import exceptions
import Timer
import Transformations

import random
from sklearn.model_selection import StratifiedKFold
import datetime



###LOAD DATA###
class Data(object):
    def __init__(self,df,type="SC",transl=None,container=None):
       self._data=df
       self._type=type
       self._lq=[]
       self.container=[]
       if container:
           self.container=container
       if transl:
           self.transl=transl
       self.calculate_group_columns_index()
       #include chromosome,position
       #self._metadata=df.filter(regex="chr|position")
       #self._data.drop(['chr','position'],axis=1)
       #self._data.replace(Settings.CODE)


    @classmethod
    #@Timer.timed
    def create_from_file(cls,filename,type,exclude=[]):
        #read header first and load only columns that are needed for the analysis
        if "USECOLS" in Config.D.keys():
          header_selection=[i for i in pd.read_csv(filename,sep="\t", nrows=1).columns if b_any(x in i for x in Config.D["USECOLS"]) and
                          not b_any(x in i for x in exclude)]
        else:
            header_selection= [i for i in pd.read_csv(filename,sep="\t", nrows=1).columns
                                                         if b_any(x in i for x in Config.D["USESAMPLES"]) and
                                                         b_any(x in i for x in Config.D["USEFEATURES"]) or
                                                         b_any(x in i for x in Config.D["USEMETADATA"]) ]
        #df = pd.read_csv(filename, sep="\t", decimal=',', index_col=[0,1,2],usecols=header_selection)
        if "NROWS" in Config.D.keys():
          df = pd.read_csv(filename, sep="\t", decimal=',',dtype={'Chr': str},index_col=[0,1,2],usecols=header_selection,nrows=Config.D['NROWS'])
        else:
            df = pd.read_csv(filename, sep="\t", decimal=',',dtype={'Chr': str},index_col=[0,1,2],usecols=header_selection)

        df.index = df.index.set_levels([df.index.levels[0],df.index.levels[1].astype(str),df.index.levels[2].astype(int)])
        df.sort_index(axis=0,inplace=True)
        #df.index.levels[1]=df.index.levels[1].astype(str)
        #df.index.levels[2]=df.index.levels[2].astype(int)
        #cols = [c for c in df.columns if not b_any(x in c for x in exclude)]
        #df=df[cols]
        tmpcol=df.columns
        df.columns=[Data.unify_columns(a) for a in df.columns.values]
        tmpdict=dict()
        for old,new in zip(tmpcol,df.columns):
            tmpdict[new]=old

        _redundand=list(df.filter(regex='\d$').columns.values)
        if _redundand:
          df.drop(df[_redundand],axis=1,inplace=True)
        #for col in df.filter(like="gtype").columns.values:
        #    df[col] = df[col].astype('category')

        to_ret=cls(df=df,type=type,transl=tmpdict)
        #to_ret.apply_NC_threshold()
        return to_ret


    def extract_chromosome(self,chromosome=1):
        return Data(self._data.xs(str(chromosome),level='Chr',axis=0),type=self._type,transl=self.transl,container=self.container)


    def loop_chromosomes(self):
        for g,d in self._data.groupby("Chr"):
            yield (g,Data(d,type=self._type,transl=self.transl,container=self.container))


    def loop_samples(self):
        for g,d in self._data.groupby(level="individual",axis=1):
            yield (g,Data(d,type=self._type,transl=self.transl,container=self.container))


    #output only features that were added during analysis-  this makes it possible to import back to GenomeStudio, headers are loaded back to its original naming
    def save_genome_studio_import_table(self,filedescr,header):
        tmp_df=self._data.loc[:,(slice(None),list(set(self.container)))]
        if header:
          individuals={k.rsplit(".",1)[0]:v.rsplit(".",1)[0] for (k,v) in self.transl.items()}
          _h=[".".join([individuals[j[0]],j[1]]) for j in tmp_df.columns.values]
          filedescr.write("\t".join(list(tmp_df.index.names) + _h)+ "\n")

        tmp_df.apply(pd.to_numeric, errors='coerce').to_csv(filedescr,sep="\t",header=False,mode="a",decimal=",")



    #output to format that is compliant with genome studio output format (Full data table)
    def save_complete_table(self,filedescr,header):
        individual_conversion={k.rsplit(".",1)[0]:v.rsplit(".",1)[0] for (k,v) in self.transl.items()}
        features_conversion= dict({k.rsplit(".",1)[1]:v.rsplit(".",1)[1] for (k,v) in self.transl.items()}.items() + {i:i for i in self.container}.items())

        if header:
          _h=[".".join([individual_conversion[j[0]],features_conversion[j[1]]]) for j in self.df.columns.values]
          #print _h
          filedescr.write("\t".join(list(self.df.index.names) + _h)+ "\n")
        self.df.to_csv(filedescr,sep="\t",header=False,mode="a")




    #this should give autosomes and sex chromosomes in 2 separate groups
    def loop_chromosomes_autosomes_XY(self):
        for g,d in self._data.groupby([v in Settings.Settings.AUTOSOMES for v in self._data.index.get_level_values("Chr")]):
            yield (g,Data(d,type=self._type,transl=self.transl,container=self.container))


    @staticmethod
    #@Timer.timed
    def unify_columns(col):
     col = re.sub(r"-|\s", "_", col)
     col=col.lower()
     if "sc" in col:
      col=re.sub(r".*(sc\d+\..+)",r"\1",col)
     if col.startswith("genera"):
      bodyid=re.sub(r"genera_(.+)\.([0-9])\.(.+)",r"\2",col)
      bodyname=Settings.Settings.CONVERSION_GENERA[bodyid]
      return \
        re.sub(r"genera_(.+)\.([0-9])\.(.+)",r"\1_" + bodyname + r".\3",col)
     else: return col

    @property
    def df(self):
        """get SC dataframe"""
        return self._data


    @df.setter
    def df(self, value):
        self._data = value

    @property
    def lqindex(self):
        """property that gives indices that didn't pass the quality check"""
        return pd.Index(self._lq)

    def add_lq_indices(self,i):
        self._lq= self._lq + i
        self._lq=list(set(self._lq))



    def _test_func(self,a):
        #a.iloc[:, a.columns.get_level_values(1)=='gtype']=a.xs('gtype',level=1,axis=1).where(a.xs('score',level=1,axis=1)>=0.15,level=1,other="NC")
        #return genotype.mask(score>=Config.D["SCORE_THRESHOLD"],other="NC")
        #if score < Config.D["SCORE_THRESHOLD"]: a.xs('gtype',level=1)="NC"
        if a.score<Config.D["SCORE_THRESHOLD"]:
            return "NC"
        else: return a.gtype

    #@Timer.timed
    def apply_NC_threshold(self):
        #self._data=self._data.groupby(level=['individual'],axis=1).apply(self._test_func
        stacked=self._data.stack(level='individual',dropna=False)
        stacked.gtype= stacked.apply(self._test_func,axis=1)
        self._data=stacked.unstack(level=-1).swaplevel(0,1,axis=1)

    #@Timer.timed
    def apply_NC_threshold_2(self):
        stacked=self._data.stack(level='individual',dropna=False)
        stacked.gtype= stacked.gtype.mask(stacked.score<Config.D["SCORE_THRESHOLD"],other="NC")
        self._data=stacked.unstack(level=-1).swaplevel(0,1,axis=1)

    #@Timer.timed
    def apply_NC_threshold_3(self,sthreshold,inplace=True):
        self._data.sort_index(axis=1,inplace=True)
        #self._data.loc[:,(slice(None),"gtype")]= self._data.loc[:,(slice(None),"gtype")].mask((self._data.loc[:,(slice(None),"score")]<Config.D["SCORE_THRESHOLD"]).values,other="NC")
        if inplace:
          self._data.loc[:,(slice(None),"gtype")]= self._data.loc[:,(slice(None),"gtype")].mask((self._data.loc[:,(slice(None),"score")]<sthreshold).values,other="NC")
          #self._data.loc[:,(slice(None),"gtype")]= self._data.loc[:,(slice(None),"gtype")].mask((self._data.loc[:,(slice(None),"score")]<sthreshold).values,other="NC")
        else:
           dc=self._data.copy()
           dc.loc[:,(slice(None),"gtype")]= dc.loc[:,(slice(None),"gtype")].mask((dc.loc[:,(slice(None),"score")]<sthreshold).values,other="NC")
           to_ret=Data(df=dc,type=self._type,transl=self.transl)
           #to_ret.apply_NC_threshold()
           return to_ret


    def apply_threshold_generic(self, sthreshold,clfname):
           dc=self._data.copy()
           clf="{}_ratio:{}_prob".format(clfname,Config.D["RATIO"])
           dc.loc[:,(slice(None),"gtype")]= dc.loc[:,(slice(None),"gtype")].mask((dc.loc[:,(slice(None),clf)]<sthreshold).values,other="NC")
           to_ret=Data(df=dc.loc[:,(slice(None),"gtype")],type=self._type,transl=self.transl)
           #to_ret.apply_NC_threshold()
           return to_ret



    def determine_genome_studio_export_threshold(self):
        return np.nanmax(self.df.loc[:,(slice(None),"score")].where((self.df.loc[:,(slice(None),"gtype")]=="NC").values).values)

    def stratify(self,n_splits=10,revert=False):
        stacked=self._data.stack(level=0)
        skf=StratifiedKFold(n_splits=n_splits)
        if not revert:
          for i,(train_index,test_index) in enumerate(skf.split(stacked, stacked["output"])):
            print len(train_index)
            yield i,Data(stacked.ix[train_index].unstack().swaplevel(1,0,axis=1)),Data(stacked.ix[test_index].unstack().swaplevel(1,0,axis=1))
        else:
           for i,(test_index,train_index) in enumerate(skf.split(stacked, stacked["output"])):
             yield i,Data(stacked.ix[train_index].unstack().swaplevel(1,0,axis=1)),Data(stacked.ix[test_index].unstack().swaplevel(1,0,axis=1))


    #self._data.filter(like="gtype").mask(~actual_filter, other="NC")
    #def custom_filter(self, s): self._data=self._data.filter(regex="chr|position|sc.*\..*")
    def custom_filter(self, s):
        samples=self._data.columns.levels[0]
        self._data=self._data.xs([i for i in samples if s in i] ,axis=1)
        #self._data=self._data.filter(regex="sc.*\..*")

    #@Timer.timed
    def slice(self, s):
        '''Create a new object from a data frame subselection -> i.e. split gDNA from different individuals
        :param s: dataset name
        :return: return new Data object with an individual
        '''
        samples=self._data.columns.levels[0]
        return Data(df=self._data.xs([i for i in samples if s in i] ,axis=1),type=self._type,transl=self.transl)

    def remove(self,s):
        samples=self._data.columns.levels[0]
        return Data(df=self._data.xs([i for i in samples if s not in i] ,axis=1),type=self._type,transl=self.transl)

    def split_up_par(self,datalist,n=0.8):

        train_data=datalist[:int((len(datalist)+1)*n)]
        test_data=datalist[int(len(datalist)*n+1):]

        self._data.loc[:,(train_data,slice(None))]
        self._data.loc[:,(test_data,slice(None))]
        #train_data = self._df[:int((len(datalist)+1)*.80)] #Remaining 80% to training set
        #test_data = data[int(len(data)*.80+1):]
        return (Data(df=self._data.loc[:,(train_data,slice(None))],type=self._type),
        Data(df=self._data.loc[:,(test_data,slice(None))],type=self._type))

    def split_up(self,n=0.8):
        '''Method splits up data into 2 halves
        :param n:
        :return:
        '''
        import copy
        datalist=copy.deepcopy(self._data.columns.get_level_values(level="individual").unique().values)
        random.shuffle(datalist)
        train_data=datalist[:int((len(datalist)+1)*n)]
        test_data=datalist[int(len(datalist)*n+1):]

        self._data.loc[:,(train_data,slice(None))]
        self._data.loc[:,(test_data,slice(None))]
        #train_data = self._df[:int((len(datalist)+1)*.80)] #Remaining 80% to training set
        #test_data = data[int(len(data)*.80+1):]
        return (Data(df=self._data.loc[:,(train_data,slice(None))],type=self._type),
        Data(df=self._data.loc[:,(test_data,slice(None))],type=self._type))

    #deprecated - slow
    def calculate_transformations(self):
        '''Calculate M and A for raw and normalized values
        :return:
        '''
        if self._type=="GDNA":
            groupfunc=Settings.Parser.parse_gdna_individual_name
        else:
            groupfunc=Settings.Parser.parse_sc_sample_name
        self.df.sort_index(level=['individual','feature'],axis=1, inplace=True)
        for g in self._data.groupby(level='individual',axis=1):
            '''g[0] - name
               g[1] - dataframe
            '''
            if g[0]!=0:
                self._data[g[0],"m"]=g[1].loc[:,(slice(None),['x','y'])].apply(Transformations.M,axis=1)
                self._data[g[0],"a"]=g[1].loc[:,(slice(None),['x','y'])].apply(Transformations.A,axis=1)
                self._data[g[0], "m_raw"]=g[1].loc[:,(slice(None),['x_raw','y_raw'])].apply(Transformations.M,axis=1)
                self._data[g[0],"a_raw"]=g[1].loc[:,(slice(None),['x_raw','y_raw'])].apply(Transformations.A,axis=1)


    def detect_outliers(self,method,features):
        if 'Chr' in features:
            features.remove('Chr')

        outliers=method.fit(self._df.loc[:,(slice(None),[features])])


    #@Timer.timed
    def calculate_transformations_2(self):
        '''Alternative - test whether its faster than original methodCalculate M and A for raw and normalized values

        :return:
        '''
        if self._type=="GDNA":
            groupfunc=Settings.Parser.parse_gdna_individual_name
        else:
            groupfunc=Settings.Parser.parse_sc_sample_name
        ##self.df.sort_index(level=['individual','feature'],axis=1, inplace=True)
        self.df.sort_index(axis=1, inplace=True)
        #this is the code making everything in 1 loop
        #y=self.df.xs("y",level="feature",axis=1)

        #x_raw=self.df.xs("x_raw",level="feature",axis=1)
        #y_raw=self.df.xs("y_raw",level="feature",axis=1)

        #pd.DataFrame(x*y,index=self.df.index,columns=pd.MultiIndex.from_tuples([(n,"test") for n in self.df.columns.get_level_values(level=0).unique()],names=self.df.columns.names))
        bucket=[]

        if "X Raw" and "Y Raw" in Config.D["USECOLS"]:
            x_raw=self.df.loc[:,(slice(None),['x_raw'])]
            y_raw=self.df.loc[:,(slice(None),['y_raw'])]

            m_raw=Transformations.M_2(x_raw,y_raw.values)
            a_raw=Transformations.A_2(x_raw,y_raw.values)

            for v,label in zip([m_raw,a_raw],["m_raw","a_raw"]):
               v.columns=pd.MultiIndex.from_tuples([(n,label) for n,f in v.columns.values],names=self.df.columns.names)
               bucket.append(v)
            self.container.append("m_raw")
            self.container.append("a_raw")
        else:
            logging.warning('Raw intensities not present in the data file')


        if "X" and "Y" in Config.D["USECOLS"] or "x" and "y" in Config.D["USECOLS"]:
            x=self.df.loc[:,(slice(None),['x'])]
            # - this is more universal approach as it works for trios as well
            # - invariant to number of index levels
            #x=self.df.xs("x",level="feature",axis=1)
            y=self.df.loc[:,(slice(None),['y'])]
            m=Transformations.M_2(x,y.values)
            a=Transformations.A_2(x,y.values)

            for v,label in zip([m,a],["m","a"]):
              v.columns=pd.MultiIndex.from_tuples([(n,label) for n,f in v.columns.values],names=self.df.columns.names)
              bucket.append(v)
            self.container.append("m")
            self.container.append("a")
        else:
            logging.warning('Normalised intensities not present in the data file')

        self._data=pd.concat(bucket + [self._data],axis=1).sort_index(axis=1)

    def _test_broadcast(self,_c,ref):
        return _c==ref

    def get_callrates_per_chr(self):
        return self.get_genotypes().groupby(level='Chr').apply(lambda s: s.apply(lambda x: pd.value_counts(x),axis=0)).unstack(level=1).stack(level=0).sort_index(axis=1)

    def get_call_rates_per_sample(self):
        valcounts=self.get_genotypes().apply(pd.value_counts,axis=0).T
        for key in Settings.Settings.CODE.keys():
            if key not in valcounts.columns:
                valcounts[key]=0
        return valcounts

    def get_call_rates_consensus(self):
        valcounts=self.consensus_genotype().value_counts()
        for key in Settings.Settings.CODE.keys():
            if key not in valcounts.index:
                valcounts[key]=0
        return valcounts

    def get_call_rates_consensus_by_chr(self):
        return self.consensus_genotype().groupby(level="Chr").value_counts().sort_index().unstack(level=1)


    def aggregate_flag_variables_by_chr(self,ref=None):
       o=self.df.loc[:,(slice(None),Config.D["FLAGVARS"])].applymap(int).groupby(level="Chr").sum().stack(level="individual")
       if ref:
         normarray_ref=ref.get_call_rates_consensus_by_chr()
         for i in Config.D['NORMARRAY']:
             if i not in normarray_ref.columns:
                 normarray_ref[i]=normarray_ref.eval(i)
         o_r=o[Config.D["FLAGVARS"]].apply(lambda x: normalize(x,normarray_ref[Config.D['NORMARRAY']]),axis=1)
         o_r.columns=pd.Index([i+"_r" for i in o_r.columns])
         return pd.concat([o,o_r],axis=1)
       o=o.assign(f=[Settings.Settings.CHR_TO_NR[str(i[0])] for i in o.index.values]).sort_values('f',kind='mergesort').drop('f',axis=1)
       return o

    def aggregate_flag_variables_by_sample(self,ref=None):
       o=self.df.loc[:,(slice(None),Config.D["FLAGVARS"])].applymap(int).sum().unstack(level=1)
       if ref:
        normarray_ref=pd.DataFrame([ref.get_call_rates_consensus()])
        for i in Config.D['NORMARRAY']:
             if i not in normarray_ref.index: normarray_ref[i]=normarray_ref.eval(i)
        o_r=o[Config.D["FLAGVARS"]]/normarray_ref[Config.D["NORMARRAY"]].squeeze().values
        o_r.columns=pd.Index([i+"_r" for i in o_r.columns])
        return pd.concat([o,o_r],axis=1)
       return o

    def aggregate_flag_variables_by_snp(self,ref=None):
        n=len(self.df.loc[:,(slice(None),'output')].columns)
        vals=((n-self.df.loc[:,(slice(None),'output')].applymap(int).sum(axis=1))-self.df.loc[:,(slice(None),'LOSS')].applymap(int).sum(axis=1))/n
        #o=self.df.loc[:,(slice(None),Config.D["FLAGVARS"])].applymap(int).sum().unstack(level=1)

        #self.df.loc[:,(slice(None),'output')].applymap(int).sum(axis=1)-self.df.loc[:,(slice(None),'LOSS')].applymap(int).sum(axis=1)
        vals.name="eval"
        return vals

    def aggregate_flag_variables(self,ref=None):
        '''
        :param ref:
        :return:
        '''
        means=self.aggregate_flag_variables_by_sample(ref).filter(regex=('.*_r')).mean()
        deviation=self.aggregate_flag_variables_by_sample(ref).filter(regex=('.*_r')).std()

        if not means.empty:
          for name,dat in zip(["mean","dev"],[means,deviation]):
              dat.index=pd.MultiIndex.from_tuples([(i,name) for i in dat.index.values], names=['data', 'stat'])
          return pd.concat([means,deviation])
        else: raise ValueError('Cannot do this type of aggregation - normalized values not present!')



    #def pos_neg_count_ado(self,ref,adi):
    #    pass

    def pos_neg_count_adi(self,ref,adi):
        homo_gdna=ref.get_call_rates_consensus()['AA']+ref.get_call_rates_consensus()['BB']
        aggs=self.aggregate_flag_variables_by_sample().mean()
        loss_homo=aggs['LOSS_AA']+aggs['LOSS_BB']
        adi_after=aggs['ADI']+aggs['ADI_AA']++aggs['ADI_BB']

        positive=homo_gdna-(loss_homo+adi_after)
        negative=adi_after

        return (positive,negative)

    def recall_ado(self,ref,ado):
        '''22/1/2018 - this is old version -
        homo_gdna=ref.get_call_rates_consensus()['AA']+ref.get_call_rates_consensus()['BB']
        aggs=self.aggregate_flag_variables_by_sample().mean()
        loss_homo=aggs['LOSS_AA']+aggs['LOSS_BB']
        ado_after=aggs['ADO']
        return (homo_gdna-(loss_homo+ado_after))/float(homo_gdna-ado)
        '''
        het_gdna=ref.get_call_rates_consensus()['AB']
        aggs=self.aggregate_flag_variables_by_sample().mean()
        loss_het=aggs['LOSS_AB']
        ado_after=aggs['ADO']
        return (het_gdna-(loss_het+ado_after))/float(het_gdna)



    def recall_adi(self,ref,adi):
        '''22/1/2018
        ab_gdna=ref.get_call_rates_consensus()['AB']
        aggs=self.aggregate_flag_variables_by_sample().mean()
        loss_ab=aggs['LOSS_AB']
        adi_after=aggs['ADI']
        return (ab_gdna-(loss_ab+adi_after))/float(ab_gdna-adi)
        '''
        homo_gdna=ref.get_call_rates_consensus()['AA']+ref.get_call_rates_consensus()['BB']
        aggs=self.aggregate_flag_variables_by_sample().mean()
        loss_homo=aggs['LOSS_AA']+aggs['LOSS_BB']
        adi_after=aggs['ADI']
        return (homo_gdna-(loss_homo+adi_after))/float(homo_gdna)

    def recall_adi_a(self,ref,adi_a):
        '''22/1/2018
        ab_gdna=ref.get_call_rates_consensus()['AB']
        aggs=self.aggregate_flag_variables_by_sample().mean()
        loss_ab=aggs['LOSS_AB']
        adi_after=aggs['ADI']
        return (ab_gdna-(loss_ab+adi_after))/float(ab_gdna-adi)
        '''
        homo_gdna=ref.get_call_rates_consensus()['BB']
        aggs=self.aggregate_flag_variables_by_sample().mean()
        loss_homo=aggs['LOSS_BB']
        adi_after=aggs['ADI_A']+aggs['ADI_AA']
        return (homo_gdna-(loss_homo+adi_after))/float(homo_gdna)

    def recall_adi_b(self,ref,adi_b):
        '''22/1/2018
        ab_gdna=ref.get_call_rates_consensus()['AB']
        aggs=self.aggregate_flag_variables_by_sample().mean()
        loss_ab=aggs['LOSS_AB']
        adi_after=aggs['ADI']
        return (ab_gdna-(loss_ab+adi_after))/float(ab_gdna-adi)
        '''
        homo_gdna=ref.get_call_rates_consensus()['AA']
        aggs=self.aggregate_flag_variables_by_sample().mean()
        loss_homo=aggs['LOSS_AA']
        adi_after=aggs['ADI_B']+aggs['ADI_BB']
        return (homo_gdna-(loss_homo+adi_after))/float(homo_gdna)




    def fpr_ado(self,ado):
        aggs=self.aggregate_flag_variables_by_sample().mean()
        return aggs['ADO']/float(ado)

    def fpr_adi(self,adi):
        aggs=self.aggregate_flag_variables_by_sample().mean()
        return aggs['ADI']/float(adi)


    def fpr_adi_a(self,adi_a):
        aggs=self.aggregate_flag_variables_by_sample().mean()
        return aggs['ADI_A']/float(adi_a)

    def fpr_adi_b(self,adi_b):
        aggs=self.aggregate_flag_variables_by_sample().mean()
        return aggs['ADI_B']/float(adi_b)

    def general_ado_adi_stat(self,final_o,X_test,Y_test,X_pred,Y_pred):
        '''this is a method moved from the main'''
        truepos_adi=len(final_o[final_o.ADI==True])
        truepos_ado=len(final_o[final_o.ADI==True])


        if Config.D["CLFNAME"]!="gaussian":
            falsepos_adi_gauss=len(final_o[(final_o.adi==True) & (final_o.pred_gauss==1)])
            falsepos_ado_gauss=len(final_o[(final_o.ado==True) & (final_o.pred_gauss==1)])


        falsepos_adi=len(final_o[(final_o.adi==True) & (final_o.pred==1)])
        falsepos_ado=len(final_o[(final_o.ado==True) & (final_o.pred==1)])

        if Config.D["CLFNAME"]!="gaussian":
            print "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (i,truepos_adi/float(len(final_o)),
                                                falsepos_adi/float(len(final_o)),
                                                falsepos_adi_gauss/float(len(final_o)),
                                                truepos_ado/float(len(final_o)),
                                                falsepos_ado/float(len(final_o)),
                                                len(final_o[final_o.pred==1])/float(len(final_o)),
                                                len(final_o[final_o.pred_gauss==1])/float(len(final_o)),
                                                (np.sum(((Y_test==True) & (Y_pred==True)))/float((sum(Y_pred==True)))))
        else:
              print "%d\t%f\t%f\t%f\t%f\t%f\t%f" % (i,truepos_adi/float(len(final_o)),
                                                falsepos_adi/float(len(final_o)),
                                                truepos_ado/float(len(final_o)),
                                                falsepos_ado/float(len(final_o)),
                                                len(final_o[final_o.pred==1])/float(len(final_o)),
                                                (np.sum(((Y_test==True) & (Y_pred==True)))/float((sum(Y_pred==True)))))



    def apply_prediction_results(self,rec=None):
      """only applies if the dataframe contains "pred" field - the prediction has been already applied

      """
      if rec==None: col="{0}_pred".format(Config.D["CLFNAME"])
      else: col=rec

      if col in self.df.columns.get_level_values(level="feature"):

      #if "pred" in self.df.columns.get_level_values(level="feature"):
         #if not type(self).__name__:
        if type(self).__name__ !="TrioLoader":
            self.df.loc[:,(slice(None),"gtype")]=self.df.loc[:,(slice(None),"gtype")].mask((self.df.loc[:,(slice(None),col)]==0).values,other="NC")
        else:
            self.df.loc[:,(slice(None),slice(None),slice(None),"gtype")]=self.df.loc[:,(slice(None),slice(None),slice(None),"gtype")].mask((self.df.loc[:,(slice(None),slice(None),slice(None),col)]==0).values,other="NC")


    def apply_proba_prediction_results(self,clfname,threshold=0.5):
      """only applies if the dataframe contains "pred" field - the prediction has been already applied
      """
      col=clfname

      if col in self.df.columns.get_level_values(level="feature"):
         #if not type(self).__name__:
         if type(self).__name__ !="TrioLoader":
            self.df.loc[:,(slice(None),"gtype")]=self.df.loc[:,(slice(None),"gtype")].mask((self.df.loc[:,(slice(None),col)]<threshold).values,other="NC")
         else:
            self.df.loc[:,(slice(None),slice(None),slice(None),"gtype")]=self.df.loc[:,(slice(None),slice(None),slice(None),"gtype")].mask((self.df.loc[:,(slice(None),slice(None),slice(None),col)]<threshold).values,other="NC")




    def apply_prediction_to_output(self):
      """ This transfers the prediction as the output variable
      :return:
      """
      if "pred" in self.df.columns.get_level_values(level="feature"): #and "output" in self.df.columns.get_level_values(level="feature"):
          #if not type(self).__name__:
          if type(self).__name__!="TrioLoader":
            if "output" in self.df.columns.get_level_values(level="feature"):
              self.df.loc[:,(slice(None),"output")]=self.df.loc[:,(slice(None),"pred")].values
            else:
              tmp=self.df.loc[:,(slice(None),"pred")].copy()
              tmp.columns=pd.MultiIndex.from_tuples([(i[0],"output") for i in tmp.columns.values],names=tmp.columns.names)
              self.df=self.df.join(tmp)
              self.df.sort_index(axis=1,inplace=True)
          else:
            if "output" in self.df.columns.get_level_values(level="feature"):
              self.df.loc[:,(slice(None),slice(None),slice(None),"output")]=self.df.loc[:,(slice(None),slice(None),slice(None),"pred")].values
            else:#there is probably no output columns, so we have to create it by copying it and then changing the index name
                tmp = self.df.loc[:,(slice(None),slice(None),slice(None),"pred")].copy()
                tmp.columns=pd.MultiIndex.from_tuples([(i[0],i[1],i[2],"output") for i in tmp.columns.values],names=tmp.columns.names)
                self.df=self.df.join(tmp)
                self.df.sort_index(axis=1,inplace=True)



            #self.df.select(lambda x: "pred" in x,axis=1)


    def update_output(self,source):
        self.df.loc[:,(slice(None),"output")]=source.df.loc[:,(slice(None),"output")].values


    def restrict_chromosomes(self,chroms_to_keep):
        self.df=self.df.loc[(slice(None),chroms_to_keep,slice(None)),:]


    def test_app_func(self,ref,sc):
        #pd.DataFrame.from_dict(Counter([(i[0],j)  for i in zip(ref_gt.values,q_gt.values) for j in i[1]]),orient='index')
        #return pd.DataFrame.from_dict(Counter([(i[0],i[1]) for i in zip(ref,sc)]),orient='index')
        #return Counter([(i[0],i[1]) for i in zip(ref,sc)])
        i=pd.MultiIndex.from_product([["AA","BB","AB","NC"], ["AA","BB","AB","NC"]])
        s = pd.Series(len(i)*[0],index=i)



        return (pd.Series((Counter([(i[0],i[1]) for i in zip(ref,sc)]))) +s).fillna(0)

    def test_func_2(self,a):
       pass


    def normalize_sc(self,a):
        return a.apply(lambda x: x*1.0/self.get_call_rates_per_sample()[a.name].fillna(0).values,axis=1)
        #return a


    def normalize_gdna(self,a,ref):
        return a*1.0/ref.get_call_rates_consensus().fillna(0)[a.name]
        #return a



    def create_transition_matrices(self,ref,normalize=True):
      ref_gt=ref.consensus_genotype()
      q_gt=self._data.loc[:,(slice(None),['gtype'])]
      transition_matrix=q_gt.apply(lambda x: self.test_app_func(ref_gt,x),axis=0)

      sc_transition_matrix=transition_matrix.groupby(level=1).apply(self.normalize_sc)
      gdna_transition_matrix=transition_matrix.groupby(level=0).apply(lambda x : self.normalize_gdna(x,ref))


      sc_transition_matrix=pd.concat([sc_transition_matrix.mean(axis=1).unstack(level=1),self.get_call_rates_per_sample().mean(axis=0).transform(lambda x: x/sum(x))],axis=1)
      gdna_transition_matrix=pd.concat([gdna_transition_matrix.mean(axis=1).unstack(level=0),ref.get_call_rates_consensus().transform(lambda x: x/sum(x))],axis=1)
      raw_tm=pd.concat([transition_matrix.mean(axis=1).unstack(level=1),ref.get_call_rates_consensus()],axis=1)

      sc_transition_matrix.columns.values[-1]="Call_rate" #sc call rate
      gdna_transition_matrix.columns.values[-1]="Call_rate"#gdna call rate
      raw_tm.columns.values[-1]="Call_rate"


      return (sc_transition_matrix,gdna_transition_matrix,raw_tm)



    #@Timer.timed
    def compare_against_reference(self,ref,func=Transformations.Equal_with_Nan,colname="output",inplace=True):
        '''
        Enrich internal dataframe by output feature
        :return:
        '''

        ref_gt=ref.consensus_genotype()
        q_gt=self._data.loc[:,(slice(None),['gtype'])]

        #this ensures that we invalidate SC calls that don't have support in gDNA - a.k.a. gDNA is NC
        #self._data.sort_index(axis=1).loc[:,(slice(None),['gtype'])].mask(ref_gt=='NC',other="NC",inplace=True)

        self._data=self._data.sort_index(axis=1)

        ###!!!THIS IS EXTREMELY IMPORTANT
        #COMMENTING THIS LINE WILL CAUSE THAT THE SC SNPS THAT ARE NOT CALLED IN GDNA ARE AUTOMATICALLY INVALIDATED -
        self._data.loc[ref_gt=="NC",(slice(None),['gtype'])]="NC"
        #self._data.sort_index(axis=1).loc[:,(slice(None),['gtype'])].mask(ref_gt=='NC',other="NC",inplace=True)
        #we comment this as we want to have complete statistics
        #self._data.sort_index(axis=1).loc[:,(slice(None),['gtype'])].mask(ref_gt=='NC',other="NC",inplace=True)

        #############################################################################################




        #Counter([(i[0],j)  for i in zip(ref_gt.values,q_gt.values) for j in i[1]])
        #print "compare_against_reference:"
        #print ref_gt.shape,q_gt.shape


        #show all transitions for gDNA==AA
        #transition_matrix.loc[(slice("AA"),slice(None))]
        #show all transitions for SC==AA
        #transition_matrix.loc[(slice(None),slice("AA"))]


        #transition_matrix.loc[(slice(None),slice("AA"))]/self.get_call_rates_per_sample()["AA"]

        #normalize transition matrix for SC
        #sc_transition_matrix=transition_matrix.loc[(slice(None),slice("AA")),:].apply(lambda x: x/self.get_call_rates_per_sample()["AA"].values,axis=1)

        #gdna_transition_matrix=transition_matrix.loc[(slice("AA"),slice(None)),:]/ref.get_call_rates_consensus()["AA"]


        #print "---------------"
        #print sc_transition_matrix
        #print gdna_transition_matrix
        #print "---------------"

        ADI_B=((q_gt=="AB")).__and__((ref_gt=="AA"),axis=0)
        ADI_A=((q_gt=="AB")).__and__((ref_gt=="BB"),axis=0)
        ADO_B=((q_gt=="AA")).__and__((ref_gt=="AB"),axis=0)
        ADO_A=((q_gt=="BB")).__and__((ref_gt=="AB"),axis=0)
        ADI_BB=((q_gt=="BB")).__and__((ref_gt=="AA"),axis=0)
        ADI_AA=((q_gt=="AA")).__and__((ref_gt=="BB"),axis=0)
        LOSS_AA=((q_gt=="NC")).__and__((ref_gt=="AA"),axis=0)
        LOSS_BB=((q_gt=="NC")).__and__((ref_gt=="BB"),axis=0)
        LOSS_AB=((q_gt=="NC")).__and__((ref_gt=="AB"),axis=0)

        LOSS=((q_gt=="NC")).__and__((ref_gt!="NC"),axis=0)
        ADO=((q_gt=="AA") | (q_gt=="BB")).__and__((ref_gt=="AB"),axis=0)
        ADI=(q_gt=="AB").__and__(((ref_gt=="AA") | (ref_gt=="BB")),axis=0)

        #vars=[ADI_B,	ADI_A,	ADO_B,	ADO_A,	ADI_BB	,ADI_AA,	LOSS_AA,	LOSS_BB,	LOSS_AB]

        #ids=["ADI_B","ADI_A","ADO_B","ADO_A","ADI_BB","ADI_AA","LOSS_AA","LOSS_BB","LOSS_AB"]

        #g=.value_counts().sort_index().unstack(level=1)[normarray]

        vars=[ADI_A,	ADI_AA,	ADI_B,	ADI_BB,	ADO_A,	ADO_B,	LOSS_AA,	LOSS_AB,	LOSS_BB, LOSS, ADO, ADI]
        ids=Config.D["FLAGVARS"]#["ADI_A",	"ADI_AA",	"ADI_B",	"ADI_BB",	"ADO_A",	"ADO_B",	"LOSS_AA",	"LOSS_AB",	"LOSS_BB"]

        self.container+=Config.D["FLAGVARS"] + ["output"]

        if len(vars)!=len(ids):
            raise ValueError('Flag variables <> labels ')

        for i,v in  zip(ids,vars):
         v.columns=pd.MultiIndex.from_tuples([(n,i) for (n,t) in v.columns.values],names=self.df.columns.names)


        #to_add=pd.DataFrame(vars)


        eq_relation=self._data.sort_index(level=['individual','feature'],axis=1).loc[:,(slice(None),['gtype'])].eq(ref.consensus_genotype(), axis=0)
        eq_relation.columns = pd.MultiIndex.from_tuples([(n,"output") for (n,t) in eq_relation.columns.values],names=self.df.columns.names)
        #ados.columns= pd.MultiIndex.from_tuples([(n,"ado") for (n,t) in ados.columns.values],names=self.df.columns.names)
        #adis.columns= pd.MultiIndex.from_tuples([(n,"adi") for (n,t) in adis.columns.values],names=self.df.columns.names)
        #adis_aa.columns= pd.MultiIndex.from_tuples([(n,"adi_aa") for (n,t) in adis.columns.values],names=self.df.columns.names)
        #adis_bb.columns= pd.MultiIndex.from_tuples([(n,"adi_bb") for (n,t) in adis.columns.values],names=self.df.columns.names)

        update=False
        if 'output' in self.df.columns.get_level_values(1).values:
           update=True

        if update:
           ret_df=pd.concat([eq_relation]+vars,axis=1).sort_index(axis=1)
           if inplace:
             #print "Update!"
             self.df.update(ret_df)
           else:
             ret_df=pd.concat([eq_relation,self.df.drop(ret_df.columns)]+vars,axis=1).sort_index(axis=1)
        else:
           ret_df=pd.concat([eq_relation,self.df]+vars,axis=1).sort_index(axis=1)
           if inplace:
             self.df=ret_df

        return ret_df
        '''
        if self._type=="GDNA":
            groupfunc=Settings.Parser.parse_gdna_individual_name
        else:
            groupfunc=Settings.Parser.parse_sc_sample_name
        '''
        '''
        for g in self._data.groupby(level='individual',axis=1):
          if g[0]!=0:
            #self._data[g[0]+".output"]=pd.concat([ref.consensus_genotype(),g[1].filter(regex="gtype")],axis=1).apply(Transformations.Equal_with_Nan,axis=1)

            tmp=pd.concat([ref.consensus_genotype(),g[1].xs('gtype',level=1,axis=1)],axis=1).apply(func,axis=1)
            if inplace:
              self._data[g[0],colname]=tmp
            else:
              data.append(tmp)
        if data:
          return pd.concat(data, axis=1, keys=[s.name for s in data])
        else: return []
        '''

        #eq_relation.columns.index = pd.MultiIndex.from_tuples(index, names = names)

    '''

    '''
    def __add__(self,other):
        a=other.df
        return 1

    def _test_apply(self,a):
        a['ADO']=1
        return a

    @Timer.timed
    def compare_against_reference_alternative(self,ref,func=Transformations.Equal_with_Nan_and_Categories,colname="output",inplace=True):
        #first columns ref genotype, the rest is SC genotypes
        #tmp=pd.concat([ref.consensus_genotype(),self._data.xs('gtype',level=1,axis=1)],axis=1)

        '''
        if self._type=="GDNA":
            groupfunc=Settings.Parser.parse_gdna_individual_name
        else:
            groupfunc=Settings.Parser.parse_sc_sample_name
        '''
        data=[]
        for g in self._data.groupby(level='individual',axis=1):
          if g[0]!=0:
            #self._data[g[0]+".output"]=pd.concat([ref.consensus_genotype(),g[1].filter(regex="gtype")],axis=1).apply(Transformations.Equal_with_Nan,axis=1)

            tmp=pd.concat([ref.consensus_genotype(),g[1].xs('gtype',level=1,axis=1)],axis=1).apply(self._compare_against_reference_alternative,axis=1)
            if inplace:
              self._data[g[0],colname]=tmp
            else:
              data.append(tmp)
        if data:
          return pd.concat(data, axis=1, keys=[s.name for s in data])
        else: return []


    def _compare_against_reference_alternative(self,ar):
        return Settings.errors.get((ar[0],ar[1]),np.nan)





    def compare(self,ref):
        if self._type=="GDNA":
            groupfunc=Settings.Parser.parse_gdna_individual_name
        else:
            groupfunc=Settings.Parser.parse_sc_sample_name

        def ado(ref):
            pass

        def adi(ref):
            pass



    def calculate_group_columns_index(self):
      if not isinstance(self._data.columns,pd.MultiIndex):
        func=Settings.Parser.parse_individual_and_feature
        self._data.columns = pd.MultiIndex.from_tuples([func(i) for i in self._data.columns.values],names=['individual','feature'])
        self.df.sort_index(level=['individual','feature'],axis=1, inplace=True)






    '''get n randomly picked up genotype samples

    '''
    #@Timer.timed
    def sample_genotype_n(self,n):
       return self._data.filter(like="gtype").sample(n,axis=1)

    '''get n randomly picked up genotype samples and return consensus

    '''
    def sample_genotype_consensus_n(self,n):
       return self._data.filter(like="gtype").sample(n,axis=1).apply(self._consensus_func, axis=1)

    def test_func(self):
        if self._type=="GDNA":
            print [(Parser.parse_feature_type(i), Parser.parse_gdna_individual_name(i), Parser.parse_gdna_sample_name(i), i) for i in self._data.columns.values]
        elif self._type=="SC":
            print [(Parser.parse_feature_type(i),Parser.parse_sc_sample_name(i),i) for i in self._data.columns.values]


    def get_individuals_count(self): return len(self.get_individuals_names())

    def get_individuals_names(self):
        '''
        :return: int
        '''
        if self._type=="GDNA":
            func=Parser.parse_gdna_individual_name
        elif self._type=="SC":
            func=Parser.parse_sc_sample_name

        return set([func(i) for i in self._data.columns.levels[0] if func(i) != 0])


    def get_feature_names(self):
        ''' Get list of features in the dataset excluding metadata
        :return: list
        '''
        func=Parser.parse_feature_type
        return list(set([func(i) for i in self._data.columns.values if func(i) != 0]))

    #@Timer.timed
    def get_samples_count(self): return len(self.get_samples_names())

    def get_samples_names(self):
        if self._type=="GDNA":
            func=Parser.parse_gdna_sample_name
        elif self._type=="SC":
            func=Parser.parse_sc_sample_name
        return set([func(i) for i in self._data.columns.get_level_values(0).values if func(i) != 0])



    #@Timer.timed
    def consensus_genotype(self):
        if not hasattr(self,'_consensus'):
          if self.get_individuals_count()>1:
            print "Warning: Heterogenious sample - more than 1 individual!"

          #if not self._lq:
          self._consensus=self._data.xs('gtype', level='feature', drop_level=False,axis=1).apply(self._consensus_func, axis=1)
          #else:
            #ind=self._data.index.difference(self._lq)
            #self._consensus=self._data.loc[ind].filter(like="gtype").apply(self._consensus_func, axis=1)
            #self._consensus=self._data.loc[ind].xs('gtype', level='feature', drop_level=False,axis=1).apply(self._consensus_func, axis=1)
          #self._consensus=self._data.xs('gtype', level='feature', drop_level=False,axis=1).apply(self._consensus_func, axis=1)
        if self._lq:
            self._consensus.loc[self._lq]="NC"
        return self._consensus


    def get_genotypes(self,inplace=False):
        '''
        '''
        if inplace:
            self._data=self._data.xs('gtype', level='feature', drop_level=True,axis=1)
            return []
        else: return self._data.xs('gtype', level='feature', drop_level=True,axis=1)

    def _consensus_func(self,ar):
      retval=ar[0]
      if len(ar)>1:
        #threshold_len=1
        threshold_len=len(ar)
        if Config.D["THRESHOLD"]!=1:
          threshold_len = floor(len(ar) * Config.D["THRESHOLD"])
        data = Counter(ar)
        mc = data.most_common(1)
        if mc[0][1] >= threshold_len:
          retval=mc[0][0]
        else:
          self._lq.append(ar.name)
          retval="NC"
      return retval


    def sample(self):
        pass

    '''in case filtering has been applied, only indices of HQ snps are reported, other the whole index
    '''
    def get_reference_snp_index(self):
        return self._data.index.difference(self._lq)

'''
def applyParallel(dfGrouped, func):
      retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
      return pd.concat(retLst)
'''

class Patterns(object):
    @staticmethod
    #@Timer.timed
    #collection of methods that operate with Data
    def check_parental_trios(mother,father,proband):
      test=pd.concat([mother.consensus_genotype(),father.consensus_genotype(),proband.consensus_genotype()],axis=1)
      return test.apply(Patterns._check_parental,axis=1)


    @staticmethod
    def check_reference(reference,sc):
        pass

    @staticmethod
    def _check_reference_n_score(ar):
        if len(ar) >=2:
           _reference=ar[0]
           _sc=ar[1:len(ar)]
           #_c=Counter([Patterns._check_parental([ar[0],ar[1],_p]) for _p in _proband])
           _c=Counter([_p==_reference for _p in _sc])
           return _c[True]/float(len(_sc))
        else:
          raise IndexError("Wrong number of individuals")

    def _check_reference_n(self,ar):
        if len(ar) >=2:
           _reference=ar[0]
           _sc=ar[1:len(ar)]
           #_c=Counter([Patterns._check_parental([ar[0],ar[1],_p]) for _p in _proband])
           return [_p==_reference for _p in _sc]
        else:
          raise IndexError("Wrong number of individuals")

    @staticmethod
    def _check_parental(ar):
      _mother=ar[0]
      _father=ar[1]
      _proband=ar[2]
      '''
      for ref_mother,ref_father,ref_proband in Settings.Settings.PARENTAL_RULES:
        if _mother == ref_mother and _father == refve_father and  _proband in ref_proband:
            return True
      return False
      '''
      return Settings.parental_rules.get((_mother,_father,_proband),False)

    @staticmethod
    def _check_parental_n(ar):
      if len(ar) >=3:
       _mother=ar[0]
       _father=ar[1]
       _proband=ar[2:len(ar)]
       #_c=Counter([Patterns._check_parental([ar[0],ar[1],_p]) for _p in _proband])
       _c=Counter([Settings.parental_rules.get((_mother,_father,_p),False) for _p in _proband])
       return _c[True]/float(len(_proband))
      else:
        raise IndexError("Wrong number of individuals")

def normalize(x,normarray):
    return x/normarray.loc[x.name[0],:].values





if __name__ == "__main__":
  '''
  gdna=Data.create_from_file(sys.argv[1],"GDNA",exclude=Settings.Settings.TO_REMOVE)
  print gdna.get_samples_count()


  sc=Data.create_from_file(sys.argv[2],"SC")
  sc.custom_filter("sc")
  mother=gdna.slice("gm07224")
  father=gdna.slice("gm07225")
  ref_proband=gdna.slice("gdna")
  #gdna.test_func()
  #sc.test_func()
  #print mother.individualsCount()
  #print gdna.samples_count()


  p=Patterns.check_parental_trios(mother,father,ref_proband)
  ref_proband.add_lq_indices(list(p[p==False].index.values))
  #sc.get_feature_names()
  sc.calculate_transformations()

  sc.compare_against_reference(ref_proband)
  #sc.df.stack(['individual','feature'],dropna=False).to_csv('individual_feature.csv')

  #extract
  #sc.df.stack(['individual'],dropna=True).to_csv('individual.csv')
  sc.df.stack(['individual'],dropna=True)[['x_raw','y_raw']]
  '''
  print datetime.datetime.now()
  #print Config.D
  Config.load(sys.argv[1])
  print Config.D
  gdna=Data.create_from_file(Config.D["GDNA"],"GDNA",exclude=Settings.Settings.TO_REMOVE)
  sc=Data.create_from_file(Config.D["SC"],"SC")
  print datetime.datetime.now()
  print "Data loaded..."


  #sc.df=sc.df.loc[(slice(None),Config.D["CHROMOSOMES"],slice(None)),:]
  #gdna.df=gdna.df.loc[(slice(None),Config.D["CHROMOSOMES"],slice(None)),:]
  sc.custom_filter("sc")
  mother=gdna.slice("gm07224")
  father=gdna.slice("gm07225")
  ref_proband=gdna.slice("gdna")
  p=Patterns.check_parental_trios(mother,father,ref_proband)
  print "Parental patterns checked..."
  ref_proband.add_lq_indices(list(p[p==False].index.values))

  tmp=[]
  #normarray=["AA","BB","AB","AB","AA","BB","AA","BB","AB"]
  normarray=Config.D["NORMARRAY"]


  #CHOICE="BY_CHR"
  CHOICE="BY_IND"


  if CHOICE=="BY_CHR":
    ref_proband.consensus_genotype().groupby(level="Chr").value_counts().sort_index().unstack(level=1).to_csv(Config.D["GDNA.CALLRATES"])
    g=ref_proband.consensus_genotype().groupby(level="Chr").value_counts().sort_index().unstack(level=1)[normarray]
  elif CHOICE=="BY_IND":
    ref_proband.consensus_genotype().value_counts().to_csv(Config.D["GDNA.CALLRATES"])
    g=ref_proband.consensus_genotype().value_counts()[normarray]


  #sc.df.loc[:,(slice(None),["ADI_B","ADI_A","ADO_B","ADO_A","ADI_BB","ADI_AA","LOSS_AA","LOSS_BB","LOSS_AB"])].applymap(int).groupby(level="Chr").sum()
  for score in xrange(0,100,5):
      print score
      t=sc.apply_NC_threshold_3(score/float(100),inplace=False)
      t.compare_against_reference(ref_proband)
      #t.df

      ####t.df.loc[:,(slice(None),["ADI_A","ADI_AA","ADI_B","ADI_BB","ADO_A","ADO_B","LOSS_AA","LOSS_AB","LOSS_BB"])].applymap(int).sum().unstack(level=1)

      #["ADI_A",	"ADI_AA",	"ADI_B",	"ADI_BB",	"ADO_A",	"ADO_B",	"LOSS_AA",	"LOSS_AB",	"LOSS_BB"]
      if CHOICE=="BY_CHR":
        t=t.df.loc[:,(slice(None),["ADI_A",	"ADI_AA",	"ADI_B",	"ADI_BB",	"ADO_A",	"ADO_B",	"LOSS_AA",	"LOSS_AB",	"LOSS_BB"])].applymap(int).groupby(level="Chr").sum().stack(level="individual")
        relative=t.apply(lambda x: normalize(x,g),axis=1)

      #relative=t.apply(lambda x: x/g.squeeze().values,axis=1)
      elif CHOICE=="BY_IND":
        t=t.df.loc[:,(slice(None),["ADI_A","ADI_AA","ADI_B","ADI_BB","ADO_A","ADO_B","LOSS_AA","LOSS_AB","LOSS_BB"])].applymap(int).sum().unstack(level=1)
        relative=t/g.values

      relative.columns=pd.Index([i+"_r" for i in relative.columns])
      #sunka=t.loc[:,(slice(None),["ADI_B","ADI_A","ADO_B","ADO_A","ADI_BB","ADI_AA","LOSS_AA","LOSS_BB","LOSS_AB"])]/ref_proband.consensus_genotype().groupby(level="Chr").value_counts().sort_index().unstack(level=1)[normarray]
      t=pd.concat([t,relative],axis=1)
      t['SCORE']=score/float(100)
      tmp.append(t)


  #sc.apply_NC_threshold_3(Config.D["SCORE_THRESHOLD"])
  #sc.calculate_transformations_2()

  #sc.compare_against_reference(ref_proband)


  result=pd.concat([d for d in tmp],axis=0)
  result.to_csv(Config.D["SC.STATS"])






  #["ADI_B","ADI_A","ADO_B","ADO_A","ADI_BB","ADI_AA","LOSS_AA","LOSS_BB","LOSS_AB"]



  #for i in xrange(0,1)

  #sc.df.groupby(level='Chr')



  '''
  working_df=test_trio.get_genotypes()

    a=working_df.groupby(level='Chr').apply(lambda s: s.apply(customfunc,axis=0)).unstack(level=1)
    errors=test_trio.df.loc[:,(slice(None),["adi","ado"])].applymap(int).groupby(level="Chr").sum()
    errors_rates=test_trio.df.loc[:,(slice(None),["adi","ado"])].applymap(int).groupby(level="Chr").apply(lambda x: x.sum()/len(x) )
    errors_rates.columns = pd.MultiIndex.from_tuples([(i,str(j)+"R") for i,j in errors_rates.columns],names=errors.columns.names)

    errors_total=test_trio.df.loc[:,(slice(None),["adi","ado"])].applymap(int).sum()
    errors_rates_total=test_trio.df.loc[:,(slice(None),["adi","ado"])].applymap(int).apply(lambda x: x.sum()/float(len(x)))
    errors_total.name="Total"
    errors_rates_total.name="Total"
    errors=errors.append(errors_total)
    errors_rates_total.index=errors_rates.columns
    errors_rates=errors_rates.append(errors_rates_total)


    b=working_df.apply(customfunc,axis=0).unstack(level=1)
    #errors.apply(customfuncErrors,axis=0)
    b.name="Total"


    writer = pd.ExcelWriter(Config.D["STATFILE"])
    output=a.append(b)

    output.columns=output.columns.droplevel(level="feature")
    output=pd.concat([output,errors,errors_rates],axis=1).sort_index(axis=1)

    #sort by chromosomes
    output=output.assign(f=[Settings.Settings.CHR_TO_NR[str(i)] for i in output.index.values]).sort_values('f',kind='mergesort').drop('f',axis=1)
    #output=output.assign(f = ideo['chrom'].apply(lambda x: convert_chr_to_number(x))).sort_values('f',kind='mergesort').drop('f', axis=1)
    #correct order of columns
    lstind=len(output.columns.levels)-1
    output=output.reindex_axis(["AA","BB","AB","NC","ABR","ABRR","TCR","nSNPs","adi","ado","adiR","adoR"],level=lstind, axis=1)
    if type=="trios":
        output=output.reindex_axis(['gdna','conc','mda','pb1','pb2','egg','ovo' ],level=2, axis=1)
    #output.columns=output.columns.reindex(output.columns.set_levels([output.columns.levels[0],output.columns.levels[1],output.columns.levels[2],["AA","BB","AB","NC","ABR","ABRR","TCR","nSNPs"]]))[0]
    output.to_excel(writer,'Sheet1',merge_cells=False)
    writer.save()
    '''





