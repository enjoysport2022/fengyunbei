# coding: utf-8
# pylint: disable = invalid-name, C0111
from __future__ import division
import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# load or create your dataset
print('Load data...')
X_train = pd.read_csv('./train.csv')
X_test = pd.read_csv('./test.csv')

id = X_test['id']
y_train = X_train['target']
X_train = X_train.drop('target',axis=1)

train_num = X_train.shape[0]
test_num = X_test.shape[0]

entire = pd.concat([X_train,X_test])
entire = entire.drop('id',axis=1)
entire.index = range(len(entire))

for c in entire.columns:
    if entire[c].dtype == 'object':
        # print(c)
#         print entire[c].nunique()
        lbl = LabelEncoder()
        lbl.fit(list(entire[c].values))
        entire[c] = lbl.transform(list(entire[c].values))

X_train = entire.loc[:train_num-1,:]
X_test = entire.loc[train_num:,:]


lgb_train = lgb.Dataset(X_train, y_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 63,
	'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# number of leaves,will be used in feature transformation
num_leaf = 63


print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100)

print('Save model...')
# save model to file
# gbm.save_model('model.txt')

print('Start predicting...')
# predict and get data on leaves, training data
y_pred = gbm.predict(X_train,pred_leaf=True)

# feature transformation and write result
print('Writing transformed training data')
transformed_training_matrix = np.zeros([len(y_pred),len(y_pred[0]) * num_leaf],dtype=np.int64)
for i in range(0,len(y_pred)):
	temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
	transformed_training_matrix[i][temp] += 1





#for i in range(0,len(y_pred)):
#	for j in range(0,len(y_pred[i])):
#		transformed_training_matrix[i][j * num_leaf + y_pred[i][j]-1] = 1

# predict and get data on leaves, testing data
y_pred = gbm.predict(X_test,pred_leaf=True)

# feature transformation and write result
print('Writing transformed testing data')
transformed_testing_matrix = np.zeros([len(y_pred),len(y_pred[0]) * num_leaf],dtype=np.int64)
for i in range(0,len(y_pred)):
	temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
	transformed_testing_matrix[i][temp] += 1



print(X_train.shape)
print(transformed_training_matrix.shape)

print(X_test.shape)
print(transformed_testing_matrix.shape)



params = {
    "objective": 'reg:logistic',
    "eval_metric":'auc',
    "seed":1123,
    "booster": "gbtree",
    "min_child_weight":5,
    "gamma":0.1,
    "max_depth": 5,
    "eta": 0.009,
    "silent": 1,
    "subsample":0.65,
    "colsample_bytree":.35,
    "scale_pos_weight":0.9
    # "nthread":16
}


df_train=xgb.DMatrix(transformed_training_matrix,y_train)
df_test = xgb.DMatrix(transformed_testing_matrix)


xgb.cv(params,df_train,nfold=5,num_boost_round=3801,early_stopping_rounds=225, verbose_eval = 50)


# import random
# seeds = []
# for i in range(0,10):
# 	seeds.append(random.randint(1,100))
# print(seeds)
#
# pred2 = np.zeros(test_num)
#
# for sd in seeds:
#     print(sd)
#     params['seed'] = sd
#     model=xgb.train(params,df_train,num_boost_round=int(2600*1.05))
#     prop = model.predict(df_test)
#     pred2 = pred2+prop
# pred2 = pred2/10
#
# submission = pd.DataFrame({ 'id': id,'predict': pred2})
# submission.to_csv("gbdtencode_xgb.csv", index=False)










'''
#for i in range(0,len(y_pred)):
#	for j in range(0,len(y_pred[i])):
#		transformed_testing_matrix[i][j * num_leaf + y_pred[i][j]-1] = 1

print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))
print('Feature importances:', list(gbm.feature_importance("gain")))


# Logestic Regression Start
print("Logestic Regression Start")

# load or create your dataset
print('Load data...')

c = np.array([1,0.5,0.1,0.05,0.01,0.005,0.001])
for t in range(0,len(c)):
	lm = LogisticRegression(penalty='l2',C=c[t]) # logestic model construction
	lm.fit(transformed_training_matrix,y_train)  # fitting the data

	#y_pred_label = lm.predict(transformed_training_matrix )  # For training data
	#y_pred_label = lm.predict(transformed_testing_matrix)    # For testing data
	#y_pred_est = lm.predict_proba(transformed_training_matrix)   # Give the probabilty on each label
	y_pred_est = lm.predict_proba(transformed_testing_matrix)   # Give the probabilty on each label

#print('number of testing data is ' + str(len(y_pred_label)))
#print(y_pred_est)

# calculate predict accuracy
	#num = 0
	#for i in range(0,len(y_pred_label)):
		#if y_test[i] == y_pred_label[i]:
	#	if y_train[i] == y_pred_label[i]:
	#		num += 1
	#print('penalty parameter is '+ str(c[t]))
	#print("prediction accuracy is " + str((num)/len(y_pred_label)))

	# Calculate the Normalized Cross-Entropy
	# for testing data
	NE = (-1) / len(y_pred_est) * sum(((1+y_test)/2 * np.log(y_pred_est[:,1]) +  (1-y_test)/2 * np.log(1 - y_pred_est[:,1])))
	# for training data
	#NE = (-1) / len(y_pred_est) * sum(((1+y_train)/2 * np.log(y_pred_est[:,1]) +  (1-y_train)/2 * np.log(1 - y_pred_est[:,1])))
	print("Normalized Cross Entropy " + str(NE))

'''