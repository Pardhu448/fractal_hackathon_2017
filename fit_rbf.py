'''Code to fit RBF on the test data'''
#####################################
'''preliminary analysis of data'''


#####################################
'''Feature extraction '''





#####################################
'''Fitting RBF on final data '''
import time

import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from math import floor 

import pandas 
import os 
os.chdir(r'/home/partha/Documents/GitFiles/projects/fractal_hackathon_2017')
train_data = pandas.read_csv(r'data/train.csv')
train_fraction = 0.7


train_data_input_sales = np.array(train_data[['Item_ID', 'Datetime', 'Category_3', 'Category_2', 'Category_1']])
train_data_sales = np.array(train_data[['Number_Of_Sales']])

svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})

X = train_data_input_sales
y = train_data_sales 
train_size = int(floor(train_fraction * len(train_data)))
print train_size 

'''
t0 = time.time()
svr.fit(X[:train_size], y[:train_size])
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit)

t0 = time.time()
kr.fit(X[:train_size], y[:train_size])
kr_fit = time.time() - t0
print("KRR complexity and bandwidth selected and model fitted in %.3f s"
      % kr_fit)

sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
print("Support vector ratio: %.3f" % sv_ratio)

t0 = time.time()
y_svr = svr.predict(X_plot)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s"
      % (X_plot.shape[0], svr_predict))

t0 = time.time()
y_kr = kr.predict(X_plot)
kr_predict = time.time() - t0
print("KRR prediction for %d inputs in %.3f s"
      % (X_plot.shape[0], kr_predict))
'''
