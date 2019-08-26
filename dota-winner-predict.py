import pandas
import numpy as np
from sklearn.metrics import roc_auc_score

data = pandas.read_csv('features.csv', index_col='match_id')
X_train = data.loc[:, 'start_time':'dire_first_ward_time']
y_train = data['radiant_win']

nan_count = X_train.isnull().sum()
nan_count = nan_count[nan_count != 0]
print(nan_count)

data_test = pandas.read_csv('features_test.csv', index_col='match_id')
X_test = data_test.loc[:, 'start_time':'dire_first_ward_time']

# print(X_test)

# pred = clf.predict_proba(X_test)[:, 1] - оценка принадлежности к 1 классу
# реализоавать AUC-ROC


#corr = np.corrcoef(features, rowvar=False)
#corr = corr.reshape((108, 108))  # 108 108
#print(np.shape(corr))
#np.savetxt('corr.csv', corr, fmt='%1.2f', delimiter=';')


