'''Code to fit RBF on the test data'''
#####################################
'''preliminary analysis of data'''
import pandas 
import os
from sklearn.preprocessing import OneHotEncoder
import numpy 
 
os.chdir(r'/home/partha/Documents/GitFiles/projects/fractal_hackathon_2017')
train_data = pandas.read_csv(r'data/train.csv')
train_fraction = 0.8

#####################################
'''Feature extraction '''
# average and std of prices for past 185 and 365 days.
# sum and std of sales for past 185 and 365 days.
def get_features_for_id(item_id_data, enc):
	item_id_data[['average_185_price']] = train_data[['Price']].rolling(185, 1).mean()
	item_id_data[['average_365_price']] = train_data[['Price']].rolling(365, 1).mean()
	item_id_data[['std_185_price']] = train_data[['Price']].rolling(185, 1).std()
	item_id_data[['std_365_price']] = train_data[['Price']].rolling(365, 1).std()
	item_id_data[['sum_185_sales']] = train_data[['Number_Of_Sales']].rolling(185, 1).sum()
	item_id_data[['sum_365_sales']] = train_data[['Number_Of_Sales']].rolling(365, 1).sum()
	item_id_data[['std_185_sales']] = train_data[['Number_Of_Sales']].rolling(185, 1).std()
	item_id_data[['std_365_sales']] = train_data[['Number_Of_Sales']].rolling(365, 1).std()
	item_id_data[['Datetime']] = pandas.to_datetime(item_id_data['Datetime'], format = '%Y-%M-%d')
	item_id_data['Qtr'] = item_id_data['Datetime'].dt.quarter 		
	item_id_data['Year'] = item_id_data['Datetime'].dt.year 		
	item_id_data['Week'] = item_id_data['Datetime'].dt.week 		
	item_id_data['Month'] = item_id_data['Datetime'].dt.month 		
	item_id_data['DoM'] = item_id_data['Datetime'].dt.day 		
	item_id_data['DoW'] = item_id_data['Datetime'].dt.dayofweek 		
	return item_id_data 

train_data = train_data.dropna()
'''To encode categorical variables using onehotencoding'''
enc = OneHotEncoder(sparse = False)
cat_data_array  = train_data[['Category_1', 'Category_2', 'Category_3']].fillna(99999).as_matrix() 
enc.fit(cat_data_array)

encoded_cols =  pandas.DataFrame(enc.transform(cat_data_array))
encoded_cols.columns = ['cat_' + str(each) for each in encoded_cols.columns]

train_data = train_data.groupby([u'Item_ID']).apply(lambda x: get_features_for_id(x, enc)).reset_index()
train_data = train_data.merge(encoded_cols, left_index = True, right_index = True)
train_data = train_data.drop([u'Category_1', u'Category_2', u'Category_3'], axis = 1)
train_data = train_data.dropna()
#print train_data.head()

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
from sklearn.metrics import mean_absolute_error

input_cols = [each for each in list(train_data.columns.values) if each not in ['index','ID','Datetime','Item_ID','Number_Of_Sales','Price']] 
train_data_input = train_data[input_cols].as_matrix()
train_data_sales = train_data[['Number_Of_Sales']].as_matrix()
train_data_prices = train_data[['Price']].as_matrix()

svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})

X = train_data_input
y = train_data_sales 
train_size = int(floor(train_fraction * len(train_data)))
print 'Training data size is - {}'.format(train_size) 
X_test = X[train_size:]
Y_test = y[train_size:]

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

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


x_test_index  = range(X_test.shape[0])
t0 = time.time()
y_svr = svr.predict(X_test)
svr_predict = time.time() - t0
print 'MAPE for SVR is - {}'.format(mean_absolute_percentage_error(Y_test, y_svr))
plt.plot(x_test_index, y_svr, color = 'b')


t0 = time.time()
y_kr = kr.predict(X_test)
kr_predict = time.time() - t0
print 'MAPE for KRR is - {}'.format(mean_absolute_percentage_error(Y_test, y_kr))
plt.plot(x_test_index, y_kr, color = 'g')
plt.plot(x_test_index, Y_test, color = 'r')





