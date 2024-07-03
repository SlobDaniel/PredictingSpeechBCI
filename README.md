# PredictingSpeechBCI
This repository contains the code and data used for predicting speech Brain-Computer Interface (BCI) performance based on patient and electrode characteristics. The study focuses on using machine learning classifiers to predict successful speech BCI outcomes from neural and spectral data derived from the open-source iBIDS dataset ‘SinglewordproductionDutch’ (Verwoert et al., 2022).

Getting Started - preprocessing. The following files contain - in order - the process of preprocessing the data to be able to build machine learning algorithmns: 
1. convert2csv      - converts the tsvfiles to csvfiles
2. add id           - adds a column with the participant id to the dataframe
3. merging_loop     - merging sub-* channels and space in a loop
4. merging_merge    - merging the above described files to one file
5. corr             - calculating correlations per participant and merging to the above descriped file
6. clean            - drops irrelavant rows of data and applies **one-hot encoding to 'description' feature**

The final dataframe 'wordprod_corr_thres' is used for analyses and predictions.

Getting Started - building models. These files use the above described data to build the LR, SVM and RF classifiers: 
1. models_thres     - building classifiers and evaluation metrics based on binary classification
2. models_bin       - building classifiers and evaluation metrics based on multiclass classification
