from json_tricks import dumps
from json_tricks import loads


D=dict()
D["THRESHOLD"] = 1
D["SCORE_THRESHOLD"]=0.15
D["FEATURES"]=["gtype", "x raw", "y raw", "score", "theta", "r", "x", "y", "b allele freq", "log r ratio"]
D["USECOLS"]=['Name',
                 'Chr',
                 'Position',
                 'GType',
                 'Score',
                 'X Raw',
                 'Y Raw',
                 'X',
                 'Y',
        ]
#D["GDNA"]="/localhome/icmm/icmm/data/data_sorted/GDNA.txt.sorted.1000"
#D["SC"]="/localhome/icmm/icmm/data/data_sorted/SC_58_SAMPLES.txt.sorted.1000"

D["GDNA"]="/localhome/icmm/icmm/LinuxServerICMM/DATA_BACKUP/GDNA.txt.sorted.1000"
D["SC"]="/localhome/icmm/icmm/LinuxServerICMM/DATA_BACKUP/SC_58_SAMPLES.txt.sorted.1000"

D["TRIOS"]="/localhome/icmm/icmm/LinuxServerICMM/all_mda_trios_score0_without_cnv_1000.txt"
D["CHROMOSOMES"]=["1"]
D["OUTLIERS"]=False
D["INPUT_FEATURE"]=["Chr","m_raw","a_raw"]
D["OUTPUT_FEATURE"]="output"
D["STATFILE"]="/localhome/icmm/icmm/LinuxServerICMM/DATA_BACKUP/default_stat.xlsx"
D["RAWPREFIX"]="default_.csv"




def save():
  json_str = dumps(D, indent=4)
  with open("CONFIGURATIONS/config.txt",'w') as f:
    f.write(json_str)

def load(s):
    with open(s,'r') as f:
        json=f.read()
        global D
        D=loads(json)
        if len(D["CHROMOSOMES"])==0:#empty list doesnt make sense, so we just generate the whole chr list
            D["CHROMOSOMES"]=[str(i) for i in xrange(1,23)] + ["X","Y","XY"]



