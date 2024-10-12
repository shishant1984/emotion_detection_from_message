import pandas as pd
import numpy as np
import pickle
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

clf = pickle.load(open('model.pkl','rb'))
test_data= pd.read_csv('./data/features/test_data_feautured.csv')

X_test= test_data.iloc[:,0:-1].values # All columsn excepy last column because its y_train
y_test= test_data.iloc[:,-1].values ## All rows of last column which is output
# Make predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metrics_dict={'accuracy': auc,
              'precision':precision,
              'recall':recall}

with open('./reports/metrics.json','w') as file:
    json.dump(metrics_dict,file,indent=4)