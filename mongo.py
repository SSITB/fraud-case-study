from class_features import Features
from pymongo import MongoClient
import pandas as pd
import numpy as np
import pprint
import pickle

mongo_client=MongoClient()
db = mongo_client['Fraud_Detection']
collection = db['Events']

def update_db():
    df2 = pd.DataFrame()
    for i in collection.find({'prob': {'$exists': False}}):
        df =  pd.DataFrame.from_dict(i, orient='index').T
        df['object_id'] = int(df['object_id'])
        df2 = df2.append(df)

    df5 = pd.DataFrame(df2.reset_index()['object_id'])
    obj = Features()
    df3 = obj.features_clean(df2)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    df3['prob'] = model.predict_proba(df3)[:,1]
    df3['prob']=pd.Series(pd.cut(df3['prob'], bins=[0.0, 0.3, 0.6, 1.0], labels=['Low Risk', 'Medium Risk', 'High Risk']))
    df3['object_id'] = df5.astype('object')
    df4 =  df3.to_dict('records')
    for i in df4:
        collection.find_one_and_update({'object_id' :i['object_id']}, {"$set": {'prob':i['prob']}})

    return collection.count_documents({"prob":'Low Risk'}), collection.count_documents({"prob":'Medium Risk'}),     collection.count_documents({"prob":'High Risk'})

low, medium, high = update_db()
