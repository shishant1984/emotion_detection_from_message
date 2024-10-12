import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml
import logging

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#fetch data from data/features files.Only training data required during training

train_data= pd.read_csv('./data/features/train_data_feautured.csv')
logger.debug("File retrived retrieved from features output")

X_train= train_data.iloc[:,0:-1].values # All columsn excepy last column because its y_train
y_train= train_data.iloc[:,-1].values ## All rows of last column which is output

params =yaml.safe_load(open('params.yaml','r'))

#Define and train on GdBoosting
clf = GradientBoostingClassifier(n_estimators=params['model_build']['n_estimators'])
clf.fit(X_train,y_train)
logger.debug("Training done")

#Save
pickle.dump(clf,open('models/model.pkl','wb')) #save model as binary file which will be used later for predictions
logger.debug("Binary file created")