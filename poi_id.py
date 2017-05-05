#!/usr/bin/python

import sys
import pickle
from time import time
sys.path.append("../tools/")
import csv
import pandas as pd
import pprint
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score

### Task 1: Select which features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
				'exercised_stock_options',
				'total_stock_value', 
				'bonus',
				'salary',
				'email_to_poi_ratio',
				'deferred_income', 
				'long_term_incentive',
				'restricted_stock', 
				'total_payments',
				'shared_receipt_with_poi',
				'loan_advances', 
				'expenses', 
				'from_poi_to_this_person',
				'other',
				'email_from_poi_ratio',
				'from_this_person_to_poi', 
				'director_fees',  
				'to_messages',  
				'deferral_payments', 
				'from_messages', 
				'restricted_stock_deferred'
				]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

### Summarize the data set
#print("Number of People: {}".format(len(data_dict)))
#print("Number of Features: {}".format(len(data_dict[data_dict.keys()[0]])))

### get # of POI
poi_cnt = 0
for person in data_dict.values():
	if person['poi'] == 1:
		poi_cnt +=1

#print("Number of POI is: {}".format(poi_cnt))

### find entries with no total_payments or total_stock_value
#for entry in data_dict:
#	person = data_dict[entry]
#	if person['total_payments'] == 'NaN' and person['total_stock_value'] == 'NaN':
#		print(entry)

### TOTAL and THE TRAVEL AGENCY IN THE PARK are not people, Eugene Lockhart is all NaN
outlier_list = ['TOTAL', 'LOCKHART EUGENE E', 'THE TRAVEL AGENCY IN THE PARK']
for outlier in outlier_list:
	data_dict.pop(outlier, 0) 


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#create new fraction to/from poi feature
for person in my_dataset.values():
	person['email_to_poi_ratio'] = 0
	person['email_from_poi_ratio'] = 0
	if float(person['from_messages']) > 0:
		person['email_to_poi_ratio'] = float(person['from_this_person_to_poi'])/float(person['from_messages'])
	if float(person['to_messages']) > 0:
		person['email_from_poi_ratio'] = float(person['from_poi_to_this_person'])/float(person['to_messages'])

### Added features by hand to features_list for kbest work
# features_list.extend(['email_to_poi_ratio', 'email_from_poi_ratio'])

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

###Do some plotting to get a feel for the data
#for point in data:
#    salary = point[1]
#    bonus = point[2]
#    matplotlib.pyplot.scatter( salary, bonus )

#plt.xlabel("salary")
#plt.ylabel("bonus")
#plt.show()

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Trying GaussianNB, SVC, DecisionTree, Kneighbors, AdaBoost

### Scale Features:
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### Select KBest
skb = SelectKBest(k = 'all')
skb.fit(features, labels)
#scores = skb.scores_
#print(scores)

### GaussianNB
### Best Results:  Acc: 0.8337, Precision: 0.3492, Recall: 0.286, F1: 0.314
# clf =  Pipeline(steps=[('scaling',scaler),("SKB", skb), ("NaiveBayes", GaussianNB())])

### SVC
### Best Results:  Acc: 0.502, Precision: 0.142, Recall: 0.546, F1: 0.2265
#clf =  Pipeline(steps=[('scaling',scaler), ("SKB", skb), ("SVC", SVC(kernel = 'linear', class_weight='balanced',
#             random_state = 42))])
#clf = SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
#	decision_function_shape=None, degree=3, gamma=0.1, kernel='sigmoid',
#	max_iter=-1, probability=False, random_state=None, shrinking=True,
#	tol=0.001, verbose=False)
### SVC GridSearch
#t0 = time()
#param_grid = {
#         'kernel': ['rbf', 'sigmoid'],
#         'class_weight': ['balanced',  ],
#          'gamma': [0.01, 0.1]
#          }
#clf = GridSearchCV(SVC(), param_grid, verbose=5, n_jobs = 2)
#clf = clf.fit(features, labels)
#print("done in %0.3fs" % (time() - t0))
#print(clf.best_estimator_)

### DecisionTree
### Best Results:  Acc: 0.8389, Precision: 0.4135, Recall: 0.4980, F1: 0.452
#clf = Pipeline(steps=[('scaling',scaler), ("SKB", skb), ("DecisionTree", DecisionTreeClassifier())])

### Neighbors
### Best Results:  Acc: 0.847, Precision: 0.258, Recall: 0.0765, F1: 0.118
#clf = Pipeline(steps=[('scaling',scaler), ("SKB", skb), ("Kneighbors", KNeighborsClassifier())])


### AdaBoost  
### Best Results:  Acc: 0.8577, Precision: 0.4495, Recall: 0.2985, F1: 0.35877 
#clf = Pipeline(steps=[('scaling',scaler), ("SKB", skb), ("AdaBoost", AdaBoostClassifier(random_state = 42))])
#clf = AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=1,
#          n_estimators=20, random_state = 21)
### AdaBoost GridSearch  
# t0 = time()
# clf = AdaBoostClassifier()
# t0 = time()
# param_grid = {'n_estimators': [10, 20, 30, 40, 50],
#                'learning_rate': [.5, .8, 1, 1.2, 1.5],
#				 'algorithm': ['SAMME.R', 'SAMME'],
#                'random_state': [21, 42, 100]}
# clf = GridSearchCV(clf, param_grid, verbose=5, n_jobs = 2)
# clf = clf.fit(features, labels)
# print("done in %0.3fs" % (time() - t0))
# print(clf.best_estimator_)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Final classifier, parameters values used were determined from GridSearchCV results.
clf = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=2,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=10,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')

### DecisionTree GridSearch
#clf = DecisionTreeClassifier()
#t0 = time()
#param_grid = {'criterion': ['gini', 'entropy'],
#				'splitter': ['best', 'random'],
#                'max_depth': [None, 2, 5, 10],
#                'min_samples_split': [2, 10, 20],
#                'min_samples_leaf': [1, 5, 10],
#                'max_leaf_nodes': [None, 5, 10, 20],
#				#'class_weight': ['balanced', ],
#				'random_state': [None, 21, 42, 100]}
#clf = GridSearchCV(clf, param_grid, verbose=5, n_jobs = 2)
#clf = clf.fit(features, labels)
#print("done in %0.3fs" % (time() - t0))
#print(clf.best_estimator_)

### Run test_classifier from tester.py to assess results
#from tester import *
#dump_classifier_and_data(clf, my_dataset, features_list)
#clf, dataset, feature_list = load_classifier_and_data()
### Run testing script
#t0 = time()
#test_classifier(clf, dataset, feature_list)
#print("Run Time: %0.3fs" % (time() - t0))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
