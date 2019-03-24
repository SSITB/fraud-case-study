import pandas as pd
from class_features import Features
from class_models import Logit, Gdbr, RandomForest, SupportVectorMachine
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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


#This function returns accuracy score and confusion matrix of a specified model
def model_score(model):
    
    model = model(X_train,y_train.values.ravel()).fit()
    y_test_predicted = model.predict(X_test)
    score = model.score(X_test,y_test) #0.95
    cm = confusion_matrix(y_test, y_test_predicted)
    return score, cm


def comparison():
        score_gdbr, cm_gdbr = model_score(Gdbr)
        score_rf, cm_rf = model_score(RandomForest)
        score_svm, cm_svm = model_score(SupportVectorMachine)
        score_logit, cm_logit = model_score(Logit)
        
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
    

