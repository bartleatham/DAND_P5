# Udacity Data Analyst NanoDegree Project 5
# Identify Fraud from Enron Email

In October of 2001, Enron, one of the largest companies in the world at the time as measured by market cap, quickly fell into bankrupcy amidst the largest financial scandal in US history.  Due to the unscrupulous behavior and business dealings of many top-level executives at the company, Enron collapsed, erasing billions of dollars of stock-holder value and employee retirement plans, all while many executives cashed in on hundreds of millions of dollars.  More reading on the Enron Scandal can be found [here](https://en.wikipedia.org/wiki/Enron_scandal).

## Purpose and Goals of this project.  
In this project, I will apply Sklearn machine learning algorithms using python on publicly available Enron financial and email datasets to create a classifier that predicts Persons of Interest (POI) in the Enron scandal.  A variety of machine learning classifiers will be experimented with, in the end I will define the final classifier and the parameters that yield the best results.  The files of interest for this project are:  
**enron61702insiderpay.pdf** - insider pay information for various Enron employees  
**poi_names.txt** - Persons of Interest in Enron scandal, these are people who were indicted, settled without admitting quilt or testified in exchange for immunity.  
**final_project_dataset.pkl** - data set with all features and labels included.  
**tester.py** - python script for performing cross validation of classifier.  
**poi.py** - main python script were features are defined, outliers addressed and classifiers experimented with and finally settled on.  
**my_feature_list.pkl** - .pkl file of final features I used.  
**my_dataset.pkl** - .pkl file of cleaned up dataset I used.  
**my_classifier.pkl** - .pkl file of my optimized classifier.  
**resources.txt** - list of webpages I used for assistance along the way.  
