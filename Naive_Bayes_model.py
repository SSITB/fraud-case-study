import pandas as pd
import numpy as np
from class_features import Features
from class_models import NaiveBayes
from bs4 import BeautifulSoup 
from sklearn.model_selection import train_test_split


df=pd.read_json('data/data.json',convert_dates=['approx_payout_date','event_created',
                                                'event_published','event_start',
                                                'event_end','user_created'])

#Features
features = Features()
X, y = features.features_clean(df)

# =============================================================================
# Naive Bayes model
# =============================================================================
def parse_text(X):
    parsed_text = []
    for idx, row in X.iteritems():
        soup = BeautifulSoup(X.loc[idx], 'html.parser')
        texts = soup.findAll(text=True)
        text_lst = list(texts)
        document = " ".join(text_lst)
        document = document.replace('\n','')
        document = document.replace('\t','')
        parsed_text.append(document)
    return parsed_text

X_parsed = parse_text(df['org_desc'])
X_parsed = np.asarray(X_parsed)

X_train, X_test, y_train, y_test = train_test_split(df['org_name'], y, 
                                                    test_size=0.20, random_state=42)

nb = NaiveBayes(X_train,y_train).fit()
y_test_predicted = nb.predict_y(X_test)

nb.score(X_test,y_test)
nb.confusion_matrix(y_test, y_test_predicted)
nb.plot_roc(X_test, y_test)
