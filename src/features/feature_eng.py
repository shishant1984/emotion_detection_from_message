import pandas as pd
import numpy as np
import os
import logging 
import yaml

from sklearn.feature_extraction.text import CountVectorizer

logger= logging.getLogger('ingestion_logs') # Logger created 
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler() # Console handler
console_handler.setLevel('DEBUG')
file_handler=logging.FileHandler('ingestions_logs.log') # File handler

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Formatter
console_handler.setFormatter(formatter) # Bind formatter to handler
file_handler.setFormatter(formatter) # Bind formatter to handler

logger.addHandler(console_handler) #Bind handler to logger
logger.addHandler(file_handler) #Bind handler to logger

#fetch data from data/preprocessed files

train_data= pd.read_csv('./data/preprocessed/train_data_processed.csv')
test_data = pd.read_csv('./data/preprocessed/test_data_processed.csv')

train_data.fillna('',inplace=True)
test_data.fillna('',inplace=True)

# Apply Bag of Words.It apply only on input

X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

max_feature_count =yaml.safe_load(open('params.yaml','r'))['feature_eng']['max_features'] # pick max_feature from params file
logger.debug('max features picked from params')

vectorizer = CountVectorizer(max_features=500) # Apply Bag of Words (CountVectorizer) for most non common words
logging.debug('Max Features retreived from params')

X_train_bow = vectorizer.fit_transform(X_train) # Fit the vectorizer on the training data and transform it
X_test_bow = vectorizer.transform(X_test) # Transform the test data using the same vectorizer

# Now joining back out test and train data after applying BOW on X only

train_df = pd.DataFrame(X_train_bow.toarray())
test_df = pd.DataFrame(X_test_bow.toarray())
train_df['label'] = y_train #Adding back output 
test_df['label'] = y_test #Adding back output 

#Store data in data/featuires

data_path= os.path.join('data','features') # define folder path data and than preprocessed inside it
os.makedirs(data_path) #create folder

train_df.to_csv(os.path.join(data_path,'train_data_feautured.csv'))
test_df.to_csv(os.path.join(data_path,'test_data_feautured.csv'))