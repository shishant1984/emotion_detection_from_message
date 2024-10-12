import pandas as pd
import numpy as np
import os
import logging
import yaml
from sklearn.model_selection import train_test_split

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


df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

df.drop(columns=['tweet_id'],inplace=True)
final_df= df[(df['sentiment']=='happiness') | (df['sentiment']=='sadness') ]


size=yaml.safe_load(open('params.yaml','r'))['data_ingestion']['test_size'] # pick size from params file
logger.debug('Test size retreived from params')

final_df['sentiment'].replace({'happiness':0,'sadness':1},inplace=True)
train_data, test_data = train_test_split(final_df, test_size=size, random_state=42)

data_path= os.path.join('data','raw') # define folder path data and than raw inside it
os.makedirs(data_path) #create folder

train_data.to_csv(os.path.join(data_path,'train.csv'))
test_data.to_csv(os.path.join(data_path,'test.csv'))