import pandas as pd
import matplotlib.pyplot as plt
from class_features import Features
from class_models import Logit, Gdbr, NaiveBayes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
from bs4 import BeautifulSoup 
import sklearn.metrics as metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import itertools
from itertools import *
from confusion_matrix import confusion_matrix_plot


df=pd.read_json('data/data.json',convert_dates=['approx_payout_date','event_created',
                                                'event_published','event_start',
                                                'event_end','user_created'])

#Features
features = Features()
X, y = features.features_clean(df)

#Checking correlations
data = X.copy()
data['fraud'] = y
correlation_matrix=data.corr()

#Train, test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)   

## =============================================================================
## Logistic regression
## =============================================================================

def logit_score():
    
    logit = Logit(X_train,y_train.values.ravel()).fit()
    y_test_predicted = logit.predict(X_test)
    score = logit.score(X_test,y_test) #0.95
    cm = confusion_matrix(y_test, y_test_predicted)
    return logit, y_test_predicted, score, cm

logit, y_test_predicted_logit, score_logit, cm_logit = logit_score()
logit.confusion_matrix_plot(y_test, y_test_predicted_logit, cmap = plt.cm.Greens)
logit.model_summary()
logit.plot_roc(X_test, y_test)

# =============================================================================
# Gradient Boosting Classifier
# =============================================================================

def gdbr_score():
    
    gdbr = Gdbr(X_train,y_train.values.ravel()).fit()
    y_test_predicted = gdbr.predict(X_test)
    score = gdbr.score(X_test,y_test) #0.98
    cm = confusion_matrix(y_test, y_test_predicted)
    return gdbr, y_test_predicted, score, cm

gdbr, y_test_predicted_gdbr, score_gdbr, cm_gdbr = gdbr_score()
gdbr.confusion_matrix_plot(y_test, y_test_predicted_gdbr, cmap = plt.cm.Greens)
gdbr.feature_importances(X_train)
gdbr.partial_dependence_plots(X_train)
gdbr.plot_roc(X_test, y_test)

# =============================================================================
# Random Forest
# =============================================================================

def rf_score():
    
    rf = RandomForestClassifier(n_estimators=1000,
                            n_jobs=-1).fit(X_train, y_train)
    y_test_predicted = rf.predict(X_test)
    score = rf.score(X_test,y_test) #0.98
    cm = confusion_matrix(y_test, y_test_predicted)
    return rf, y_test_predicted, score, cm

rf, y_test_predicted_rf, score_rf, cm_rf = rf_score()
confusion_matrix_plot(cm_rf, cmap=plt.cm.Greens)

# =============================================================================
# SVM
# =============================================================================

def scaling(cont_vars, X_train, X_test):
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    X_train_copy[cont_vars]=StandardScaler().fit_transform(X_train_copy[cont_vars])
    X_test_copy[cont_vars]=StandardScaler().fit_transform(X_test_copy[cont_vars])
    return X_train_copy, X_test_copy
    
#Standardizing/Rescaling continuous variables
cont_vars=['body_length', 'name_length', 'sale_duration', 'user_age',
        'org_facebook','org_twitter', 'avg_ticket_cost','tot_ticket_quant']
X_train_svm, X_test_svm = scaling(cont_vars, X_train, X_test)

def svm_score():
    
    svm = SVC(gamma='scale',probability=True).fit(X_train_svm,y_train)
    y_test_predicted = svm.predict(X_test_svm)
    score = svm.score(X_test_svm,y_test) 
    cm = confusion_matrix(y_test, y_test_predicted)
    return svm, y_test_predicted, score, cm

svm, y_test_predicted_svm, score_svm, cm_svm = svm_score()
confusion_matrix_plot(cm_svm, cmap=plt.cm.Greens)

# =============================================================================
# ROC curve
# =============================================================================

def roc_comparison():
    #Logit
    preds_logit = logit.predict_proba(X_test)[:,1]
    fpr_logit, tpr_logit, threshold = metrics.roc_curve(y_test, preds_logit)
    roc_auc_logit = metrics.auc(fpr_logit, tpr_logit)
    
    #Gdbr
    preds_gdbr = gdbr.predict_proba(X_test)[:,1]
    fpr_gdbr, tpr_gdbr, threshold = metrics.roc_curve(y_test, preds_gdbr)
    roc_auc_gdbr = metrics.auc(fpr_gdbr, tpr_gdbr)
    
    #Random Forest
    preds_rf = rf.predict_proba(X_test)[:,1]
    fpr_rf, tpr_rf, threshold = metrics.roc_curve(y_test, preds_rf)
    roc_auc_rf = metrics.auc(fpr_rf, tpr_rf)
    
    #SVM
    preds_svm = svm.predict_proba(X_test_svm)[:,1]
    fpr_svm, tpr_svm, threshold = metrics.roc_curve(y_test, preds_svm)
    roc_auc_svm = metrics.auc(fpr_svm, tpr_svm)
    
    # method I: plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_gdbr, tpr_gdbr, 'r', label = 'AUC Gradient Boost = %0.2f' % roc_auc_gdbr)
    plt.plot(fpr_rf, tpr_rf, 'g', label = 'AUC Random Forest = %0.2f' % roc_auc_rf)
    plt.plot(fpr_svm, tpr_svm, 'y', label = 'AUC SVM = %0.2f' % roc_auc_svm)
    plt.plot(fpr_logit, tpr_logit, 'b', label = 'AUC Logit = %0.2f' % roc_auc_logit)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    # Zoomed graph
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_gdbr, tpr_gdbr, 'r', label = 'AUC Gradient Boost = %0.2f' % roc_auc_gdbr)
    plt.plot(fpr_rf, tpr_rf, 'g', label = 'AUC Random Forest = %0.2f' % roc_auc_rf)
    plt.plot(fpr_svm, tpr_svm, 'y', label = 'AUC SVM = %0.2f' % roc_auc_svm)
    plt.plot(fpr_logit, tpr_logit, 'b', label = 'AUC Logit = %0.2f' % roc_auc_logit)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.ylim(0.9, 1)
    plt.xlim(0, 0.6)
    plt.show()


#Results comparison
def score_comparison():
        logit, y_test_predicted_logit, score_logit, cm_logit = logit_score()
        gdbr, y_test_predicted_gdbr, score_gdbr, cm_gdbr = gdbr_score()
        rf, y_test_predicted_rf, score_rf, cm_rf = rf_score()
        svm, y_test_predicted_svm, score_svm, cm_svm = svm_score()

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
    roc_comparison()
    score_comparison()
    



