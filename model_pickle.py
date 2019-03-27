# =============================================================================
# Gradient Boosting Classifier pickle file
# =============================================================================

import pandas as pd
from class_features import Features
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_json('data/data.json',convert_dates=['approx_payout_date','event_created',
                                                'event_published','event_start',
                                                'event_end','user_created'])


import pickle
if __name__ == '__main__':
    features = Features()
    X = features.features_clean(df)
    
    #Target variable = 1 if the event is fraudulent
    df['fraud'] = df['acct_type'].str.contains('fraud')
    y = pd.get_dummies(df['fraud'],drop_first=True)
    
    gdbr = GradientBoostingClassifier(learning_rate=0.01,
                                  max_depth=10,
                                  n_estimators=1000,
                                  min_samples_leaf=100,
                                  max_features=4)
    
    gdbr.fit(X, y.values.ravel())

    with open('model.pkl', 'wb') as f:
        pickle.dump(gdbr, f)
        