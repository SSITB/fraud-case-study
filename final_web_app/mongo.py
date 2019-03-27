from class_features import Features
from pymongo import MongoClient
import pandas as pd

mongo_client=MongoClient()
db = mongo_client['Fraud_Detection']
collection = db['Events']


def update_db(model):
    
    new_events = pd.DataFrame()
    
    for i in collection.find({'prob': {'$exists': False}}):
        event =  pd.DataFrame.from_dict(i, orient='index').T
        event['object_id'] = int(event['object_id'])
        new_events = new_events.append(event)
    
    new_events = new_events.reset_index()
    
    features = Features()
    X = features.features_clean(new_events)

    X['prob'] = model.predict_proba(X)[:,1]
    X['prob']=pd.Series(pd.cut(X['prob'], bins=[0.0, 0.3, 0.6, 1.0], 
                             labels=['Low Risk', 'Medium Risk', 'High Risk']))
    X['object_id'] = pd.DataFrame(new_events['object_id']).astype('object')
    X_dict =  X.to_dict('records')
    for i in X_dict:
        collection.find_one_and_update({'object_id' :i['object_id']},
                                        {"$set": {'prob':i['prob']}})

    return None
    
def pull_values():

    return collection.count_documents({"prob":'Low Risk'}), \
            collection.count_documents({"prob":'Medium Risk'}), \
                collection.count_documents({"prob":'High Risk'})



