# SureTypeSC
SureTypeSC is implementation of algorithm for regenotyping of single cell data coming from Illumina BeadArrays. 

## Getting Started

These instructions will guide you through installation and testing of the program. The large project data (classifiers and testing data) are storted on Google Drive due to the GitHub filesize limitation. Please refer to the links in the corresponding folders (clf/ and data/).

### Prerequisites
* python 2 (tested on Python 2.7.5)
* scikit >= v0.19.1 (http://scikit-learn.org/stable/)
* numpy >= v1.14.1 (http://www.numpy.org/)
* pandas >= v0.22.0 (https://pandas.pydata.org/)
* json_tricks >= v3 (https://pypi.org/project/json_tricks/)

### Installing

All the required packages (except for git-lfs) should be automatically installed during the installation procedure (tested on Red Hat EL7, Centos and Windows 7 - all 64bit versions)


To download the depository:
```
git clone https://github.com/puko818/SureTypeSC
```
and install:
```
cd SureTypeSC
python setup.py install
```
now unzip classifiers:
```
unzip "clf/*.zip" -d clf
```

## Running the program - basic configuration
All the parameters necessary for running the program are stored in the configuration file. The configuration files are stored in folder config/.
The classification of the genotypes is always run separately for autosomes and sex chromosomes (if selected for analysis).
To run test the program on small dataset with a minimized random forest model (trained on around ~50,000 SNPs), run:
```
python genotyping/SureTypeSC_basic_v02.py config/GM12878_basic_test.conf
```
The dataset will undergo Random Forest analysis and then SureTypeSC create  separate Gaussian discriminant models for autosomes and sex chromosomes. 
SureTypeSC_basic.py creates 3 files, they will be stored in the output/ folder:
* \*-fulltable.txt - original datable from GenomeStudio (input dataset)  enriched by predictions from SureTypeSC
* \*-gsimport.txt -  datatable only with features created by SureTypeSC - can be directly imported back to GenomeStudio via Subcolumns import 
* \*-log.txt log of the events

depending on the mode selected in the configuration file (mode for standard SC genotyping, high precision or high recall), the program furthermore generates 1-3 files with single-cell genotypes:
* \*SC_genotypes_standard.txt - standard SC genotyping using RF-GDA with balanced recall and precision
* \*SC_genotypes_precision.txt
* \*SC_genotypes_recall.txt

The program enriches every sample in the input data by :

| Subcolumn name  | Meaning |
| ------------- | ------------- |
| rf_ratio:1_pred  | Random Forest prediction (binary)  |
| rf_ratio:1_prob  | Random Forest Score for the positive class |
| gda_ratio:1_prob | Gaussian Discriminant Analysis score for the positive class  | 
| gda_ratio:1_pred | Gaussian Disciminant Analysis prediction (binary) | 
| rf-gda_ratio:1_prob | combined 2-layer RF and GDA - probability score for the positive class | 
| rf-gda_ratio:1_pred | binary prediction of RF-GDA | 

These columns are visible in \*-fulltable.txt and \*-gsimport.txt. Ratio 1 means the classifier was trained on balanced dataset (which is a default option)

## Running the program - configuration with the full dataset
* run with  deidentified dataset GM12878 and RF classifier trained with 30 trees in the forest on the full GM7228 dataset (analysis will take ~1hour)

## Running the program on own datasets
* when exporting the data from the GenomeStudio, we recommend to change the GenCall score from 0.15 to 0.01. This allows for higher recall as GenCall generally behaves suboptimal in the single cell environment (Vogel et al., 2019). 
```
python genotyping/SureTypeSC_basic_v02.py config/GM12878_full_test.conf
```


<!---## Running the program - validation--->
<!--- Validation procedures are implemented in SureTypeSC.py. To run a validation procedure equivalent to basic configuration, run:--->
<!---```--->
<!---python genotyping/SureTypeSC.py config/GM12878_basic_test.conf--->
<!---```--->


### Contact
In case of any questions please contact Ivan Vogel (ivogel@sund.ku.dk)

