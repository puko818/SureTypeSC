
'''
Module that encapsulates all the settings,constants and parsers
Author: Ivan Vogel
'''


import re
from collections import defaultdict
from json_tricks import dumps
from json_tricks import loads
import numpy as np


parental_rules=defaultdict(lambda: False)

parental_rules=defaultdict.fromkeys([
    ("AA","AA","AA"),
    ("AA","AB","AA"),
    ("AA","AB","AB"),
    ("AA","BB","AB"),
    ("AA","BB","AB"),
    ("BB","AA","AB"),
    ("BB","AB","AB"),
    ("BB","AB","BB"),
    ("BB","BB","BB"),
    ("AB","AA","AB"),
    ("AB","AA","AA"),
    ("AB","BB","AB"),
    ("AB","BB","BB"),
    ("AB","AB","AB"),
    ("AB","AB","BB"),
    ("AB","AB","AA"),
],True)




errors=dict({
    ('AA','AB'):'ADI',
    ('BB','AB'):'ADI',
    ('AB','AA'): 'ADO',
    ('AB','BB'): 'ADO'
}
)

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Settings(object):
   AUTOSOMES = ["1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                "21",
                "22"]
   SEX = ["X",
          "Y",
          "XY"
   ]
   CODE = {"AA": 0, "BB": 1, "AB": 2, "NC": 3}
   DECODE={0: "AA", 1: "BB", 2: "AB", 3: "NC"}
   CHR_CODE = {"X": "23", "Y": "24", "XY": "25"}
   CONVERSION_GENERA={"1": "pb1", "2": "pb2", "3": "egg"}
   PARENTAL_RULES=\
        [["AA","AA",("AA")],
         ["BB","BB",("BB")],
         ["AB","AB",("AA","AB","BB")],
         ["AA","AB",("AA","AB")],
         ["AB","AA",("AA","AB")],
         ["BB","AB",("BB","AB")]]
   TO_REMOVE=["GM07226_8_3",
               "GM07226_8_1",
               "GM07226_8_2",
               "GM07225_8_3",
               "GM07225_8_1",
               "GM07225_8_2"]


   CHR_TO_NR={ "1": 1,
           "2": 2,
           "3": 3,
           "4": 4,
           "5": 5,
           "6": 6,
           "7": 7,
           "8": 8,
           "9": 9,
           "10": 10,
           "11": 11,
           "12": 12,
           "13": 13,
           "14": 14,
           "15": 15,
           "16": 16,
           "17": 17,
           "18": 18,
           "19": 19,
           "20": 20,
           "21": 21,
           "22": 22,
           "X":23,
           "Y":24,
           "XY":25,
           "Total": 26}

   BFREQ={"AA":0.0,
          "AB":0.5,
          "BB":1.0,
          "NC":np.nan
          }


   BFREQ_HOMO_HETERO={"AA":0.0,
          "AB":0.5,
          "BB":0.0,
          "NC":np.nan
          }

   COLOURS={
       "0.0": "255,215,0",
       "1.0": "0,128,0",
       "0.5": "255,0,0"
   }


   HETSAMPLES=["pb1","gdna"]
   HOMOSAMPLES=["egg","pb2"]

   COMPONENTS={
               "AA_true":("AA",1),
               "AA_false":("AA",0),
               "BB_true":("BB",1),
               "BB_false":("BB",0),
               "AB_true":("AB",1),
               "AB_false":("AB",0)
               }




class Parser(object):
    @staticmethod
    def parse_gdna_individual_name(s):
        '''

        :param s: string
        :return: return SC sample name
        '''
        r = re.search(r"(.+?)_.+\.", s)
        if r:
         return r.group(1)
        else:
         return 0


    @staticmethod
    def parse_gdna_sample_name(s):
        '''

        :param self:
        :return:
        '''
        #.+(\d)+\..+$
        r = re.search(r"(.+)\.{0,1}", s)
        if r:
         return r.group(1)
        else:
         return 0


    @staticmethod
    def parse_sc_sample_name(s):
        '''
        :param s: string
        :return: return SC sample name
        '''
        #TODO

        #r = re.search(r".*([s|g]c\d+)\.{0,1}.*", s)
        #NOTE adjusted 14.6.2018 to work with GM12878
        r= re.search(r".*([s|g]c[_]{0,1}\d+)\.{0,1}.*", s)
        if r:
         return r.group(1)
        else:
         return 0

    @staticmethod
    def parse_feature_type(s):
        '''
        :param s: string
        :return: feature name
        '''
        r = re.search(r".*\.(.+?)$", s)
        if r:
         return r.group(1)
        else:
         return 0

    @staticmethod
    def parse_individual_and_feature(s):
        r=re.search(r"(.+)\.(.+)",s)
        if r:
           return ((r.group(1),r.group(2)))
        else:
           return (("Metadata",s))

    @staticmethod
    def parse_individual_sampleid_celltype_feature(s):
        r=re.search(r"(.+)\.(.+)",s)
        if r:
           feature=r.group(2)
           individual_sampleid_celltype=r.group(1).split("_")
           if len(individual_sampleid_celltype)!=3:
               if len(individual_sampleid_celltype)==2:
                   individual_sampleid_celltype.append(individual_sampleid_celltype[-1])
           if individual_sampleid_celltype[1]==individual_sampleid_celltype[2] and "." in individual_sampleid_celltype[1]:
               return (individual_sampleid_celltype[0],
                       individual_sampleid_celltype[1].split(".")[0],
                       individual_sampleid_celltype[2].split(".")[1],
                       r.group(2))
           else : return ((individual_sampleid_celltype[0],individual_sampleid_celltype[1],individual_sampleid_celltype[2],r.group(2)))
        else:
           return (("Metadata",s))



    @staticmethod
    def parse_individual_sampleid_subtype_and_feature(s):
        r=re.match(r"(?P<individual>.+)[_|.](?P<celltype>.+)\.(?P<feature>.+)",s)
        t=r.groupdict()
        _in=re.split(r'\.|_', t['individual'])
        individual=_in[0]
        sampleid=None
        if (len(_in)>1):
            individual=".".join(_in[:-1])
            sampleid=_in[-1]
        if t:
           return ((individual,sampleid,t['celltype'],t['feature']))
        else:
           return (("Metadata","","",s))

if __name__ == "__main__":
    s=Settings()
    save(s)

    d=load("config.txt")
    pass
    #members = [attr for attr in dir(s) if not callable(getattr(s, attr)) and not attr.startswith("__")]
    #print members
