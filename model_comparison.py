import pandas as pd
from class_features import Features
from class_models import Logit, Gdbr
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

df=pd.read_json('data/data.json',convert_dates=['approx_payout_date','event_created',
                                                'event_published','event_start',
                                                'event_end','user_created'])

#Features
features = Features()
X, y = features.features_clean(df)

#Train, test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)   

## Logistic regression
def logit_score():
    
    logit = Logit(X_train,y_train.values.ravel()).fit()
    y_test_predicted = logit.predict(X_test)
    score = logit.score(X_test,y_test) #0.95
    cm = confusion_matrix(y_test, y_test_predicted)
    return score, cm

# Gradient Boosting Classifier
def gdbr_score():
    
    gdbr = Gdbr(X_train,y_train.values.ravel()).fit()
    y_test_predicted = gdbr.predict(X_test)
    score = gdbr.score(X_test,y_test) #0.98
    cm = confusion_matrix(y_test, y_test_predicted)
    return score, cm

# Random Forest
def rf_score():
    
    rf = RandomForestClassifier(n_estimators=1000,
                            n_jobs=-1).fit(X_train, y_train)
    y_test_predicted = rf.predict(X_test)
    score = rf.score(X_test,y_test) #0.98
    cm = confusion_matrix(y_test, y_test_predicted)
    return score, cm

# SVM
def scaling(cont_vars, X_train, X_test):
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    X_train_copy[cont_vars]=StandardScaler().fit_transform(X_train_copy[cont_vars])
    X_test_copy[cont_vars]=StandardScaler().fit_transform(X_test_copy[cont_vars])
    return X_train_copy, X_test_copy
    

def svm_score():
    
    #Standardizing/Rescaling continuous variables
    cont_vars=['body_length', 'name_length', 'sale_duration', 'user_age',
        'org_facebook','org_twitter', 'avg_ticket_cost','tot_ticket_quant']
    X_train_svm, X_test_svm = scaling(cont_vars, X_train, X_test)
    
    svm = SVC(gamma='scale',probability=True).fit(X_train_svm,y_train)
    y_test_predicted = svm.predict(X_test_svm)
    score = svm.score(X_test_svm,y_test) 
    cm = confusion_matrix(y_test, y_test_predicted)
    return score, cm


def comparison():
        score_gdbr, cm_gdbr = gdbr_score()
        score_rf, cm_rf = rf_score()
        score_svm, cm_svm = svm_score()
        score_logit, cm_logit = logit_score()
        
        scores = [score_gdbr, score_rf, score_svm, score_logit]
        cms = [cm_gdbr, cm_rf, cm_svm, cm_logit]
        names = ['Gradient Boosting', 'Random Forest', 'SVM', 'Logit']
        
        for i in range(len(scores)):
            text = names[i]+' accuracy score: '
            print(text.ljust(34),scores[i])
        
        print(' ')
        
        for i in range(len(cms)):
            TN = cms[i][0][0]
            FP = cms[i][0][1]
            FN = cms[i][1][0]
            TP = cms[i][1][1]
            print(names[i].ljust(17), 'TN:', TN, 'FP:', FP, 'FN:', str(FN).ljust(3), 'TP:', TP)
    

if __name__ == "__main__":
    comparison()
    

