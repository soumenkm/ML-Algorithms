#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 11:51:15 2024

@author: soumensmacbookair
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
from sklearn import inspection

#%% Import Datasets
iris_ds = datasets.load_iris()
"""
iris_ds.__dir__()
dict_keys(['data', 'target', target_names', 'DESCR', 'feature_names'])
"""

data_x = np.array(iris_ds["data"], dtype=np.float32)
target_y = np.array(iris_ds["target"], dtype=np.int64)
target_class_labels = np.array(iris_ds["target_names"], dtype=np.str_)
feature_names = np.array(iris_ds["feature_names"], dtype=np.str_)

#%% Preprocess dataset
scalar = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
scalar.fit(data_x) # scales the data_x data by mean and variance
scalar.n_features_in_ # number of features seen during scaling
scalar.n_samples_seen_ # number of samples seen during scaling
scalar.mean_ # mean of all the features used for scaling
scalar.var_ # variance of all the features used for scaling

data_x_scaled = scalar.transform(data_x, copy=True) # apply scaling
data_x_original = scalar.inverse_transform(data_x_scaled, copy=True) # remove scaling

#%% Split the dataset
train_x, test_x, train_y, test_y = model_selection.train_test_split(data_x_scaled, target_y, test_size=0.2, train_size=0.8,
                                                                    random_state=1, shuffle=True) # size can be number of examples

#%% Train the logistic regression model
logreg = linear_model.LogisticRegression(penalty="l2",
                                         solver="sag",
                                         max_iter=100,
                                         random_state=1,
                                         tol=1e-4, # stopping criteria: max abs change in parameters wrt previous epoch
                                         C=1.0, # C is inversely prop to reguralization strength. Large C means less reguralization
                                         verbose=2,
                                         multi_class="multinomial")

logreg.fit(train_x[:,:2], train_y) # train the classifier using training data

logreg.classes_ # class labels known to the classifier
logreg.coef_ # weights (n_classes, n_features)
logreg.intercept_ # bias (n_classes,)
logreg.n_features_in_ # number of features seen during fit
logreg.n_iter_ # number of epochs used during training

#%% Predict from the logistic regression model
logreg.predict(test_x[0:1,:2]) # returns predicted class labels
logreg.predict_proba(test_x[0:1,:2]) # returns softmax probabilities
logreg.decision_function(test_x[0:1,:2]) # returns z score before applying softmax
logreg.score(test_x, test_y) # returns the accuracy of the test samples

#%% Plot the decision boundary
display = inspection.DecisionBoundaryDisplay.from_estimator(estimator=logreg,
                                                            X=train_x[:,:2],
                                                            xlabel=feature_names[0],
                                                            ylabel=feature_names[1],
                                                            alpha=0.5,
                                                            levels=[-0.5, 0.5, 1.5, 2.5],
                                                            colors=["purple", "green", "blue"])

display.ax_.scatter(train_x[:, 0][train_y == 0], train_x[:, 1][train_y == 0], c="purple",
                    label=target_class_labels[0])
display.ax_.scatter(train_x[:, 0][train_y == 1], train_x[:, 1][train_y == 1], c="green",
                    label=target_class_labels[1])
display.ax_.scatter(train_x[:, 0][train_y == 2], train_x[:, 1][train_y == 2], c="blue",
                    label=target_class_labels[2])
display.ax_.set_title("Decision Boundary of Logistic Regression")
display.ax_.legend()





