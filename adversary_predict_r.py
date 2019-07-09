from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from numpy import genfromtxt
import math

# 2-0: crawler
# 5-1: R
# 16-2: d_sample
# 18-3: e_sample
crawler = "mod"

# my_data = genfromtxt('./log/aggregate-r-properties.txt', delimiter=',', dtype=str, usecols=(2, 5, 16, 18), autostrip=True)
# filter_data = my_data[my_data[:, 0] == crawler]
# training_X = filter_data[:, np.newaxis, 2].astype(float)
# training_Y = filter_data[:,1].astype(float)
#
# # 2-0: crawler
# # 5-1: R
# # 16-2: d_sample
# # 19-3: e_sample
# my_data_testing = genfromtxt('./log/aggregate-r-testing-properties.txt', delimiter=',', dtype=str, usecols=(2, 5, 16, 19), autostrip=True)
# # my_data_testing = genfromtxt('./log/aggregate-r-testing-test.txt', delimiter=',', dtype=str, usecols=(3,4,5,6), autostrip=True)
# #
# # filter_data_testing = my_data_testing
# filter_data_testing = my_data_testing[my_data_testing[:, 0] == crawler]
#
# testing_X = filter_data_testing[:, np.newaxis, 2].astype(float)
# # testing_X = filter_data_testing[:,  [2,3]].astype(float)
# testing_Y = filter_data_testing[:,1].astype(float)

# 0: p, 1: crawler, 2: dataset, 3: R,
# 4: d_samp_p, 5: e_samp_p, 6: cc_samp_p, 7: d*cc, 8:e*cc
feature =  [0, 4, 5, 6, 7]
my_data = genfromtxt('./log/sample-training-aggregate.txt', delimiter=',', dtype=str, autostrip=True, skip_header=1)

filter_data = my_data[my_data[:, 1] == crawler]
filter_data = my_data
training_X = filter_data[:, feature].astype(float)
training_Y = filter_data[:,3].astype(float)
print(training_Y)


my_data = genfromtxt('./log/sample-testing-aggregate.txt', delimiter=',', dtype=str, autostrip=True, skip_header=1)
filter_data = my_data[my_data[:, 1] == crawler]
filter_data = my_data
testing_X = filter_data[:, feature].astype(float)
testing_Y = filter_data[:,3].astype(float)




regr = linear_model.LinearRegression(normalize=True)

# Train the model using the training sets
regr.fit(training_X, training_Y)

# Make predictions using the testing set
predict_Y = regr.predict(testing_X)

# The coefficients
print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_ )
# The mean squared error
print("Mean squared error: %.5f"
      % mean_squared_error(testing_Y, predict_Y))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % r2_score(testing_Y, predict_Y))

print('Variance score: %.4f' % explained_variance_score(testing_Y, predict_Y))
print('Variance score: %.4f' % explained_variance_score(testing_Y, predict_Y, multioutput='variance_weighted'))


# # # Plot outputs
# plt.scatter(testing_X, testing_Y,  color='black')
# plt.plot(testing_X, predict_Y, color='blue', linewidth=3)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()