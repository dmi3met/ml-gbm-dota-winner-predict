import pandas
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

data = pandas.read_csv('features.csv', index_col='match_id')
X_train = data.loc[:, 'start_time':'dire_first_ward_time']
y_train = data['radiant_win']

nan_count = X_train.isnull().sum()
nan_count = nan_count[nan_count != 0]
print('Признаки с пропусками (NaN) |  Количество пропусков')
print(nan_count)

X_train.fillna(inplace=True, value=0)

print('Столбец с целевой переменной - radiant_win')

data_test = pandas.read_csv('features_test.csv', index_col='match_id')
X_test = data_test.loc[:, 'start_time':'dire_first_ward_time']
X_test.fillna(inplace=True, value=0)

gbm = GradientBoostingClassifier(n_estimators=250, verbose=True,
                                 random_state=241, learning_rate=lr)
gbm.fit(X_train, y_train)

# print(X_test)

# pred = clf.predict_proba(X_test)[:, 1] - оценка принадлежности к 1 классу
# реализоавать AUC-ROC


#corr = np.corrcoef(features, rowvar=False)
#corr = corr.reshape((108, 108))  # 108 108
#print(np.shape(corr))
#np.savetxt('corr.csv', corr, fmt='%1.2f', delimiter=';')


