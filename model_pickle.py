# =============================================================================
# Gradient Boosting Classifier pickle file
# =============================================================================

import pandas as pd
from class_features import Features
from class_models import Gdbr

df=pd.read_json('data/data.json',convert_dates=['approx_payout_date','event_created',
                                                'event_published','event_start',
                                                'event_end','user_created'])

import pickle
if __name__ == '__main__':
    features = Features()
    X, y = features.features_clean(df)
    gdbr = Gdbr(X, y.values.ravel()).fit()

    with open('model.pkl', 'wb') as f:
        pickle.dump(gdbr, f)
        