import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler



class Logit():
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def fit(self):
        self.model = LogisticRegression(C=1000)
        self.model.fit(self.X, self.y)
        return self
    
    def predict(self, X):
        y_predicted = self.model.predict(X)
        return y_predicted
    
    def predict_proba(self, X):
        y_proba = self.model.predict_proba(X)
        return y_proba
    
    def score(self, X, y):
        score = self.model.score(X, y)
        return score
    
    def confusion_matrix_plot(self, y, y_predicted,cmap):
        cm = confusion_matrix(y, y_predicted)
        plt.clf()
        plt.imshow(cm, cmap, interpolation='nearest')
        classNames = ['Not Fraud','Fraud']
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        thresh = cm.max() / 2.
        s = [['TN','FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+"\n"+str(cm[i][j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
        plt.show()
        
    def confusion_matrix(self, y, y_predicted):
        return confusion_matrix(y, y_predicted).ravel()
        
    def model_summary(self):
        X = sm.add_constant(self.X)
        model = sm.Logit(self.y,X.astype(float))
        model=model.fit()
        return model.summary()
    
    def plot_roc(self, X, y):
        y_predicted_proba = self.model.predict_proba(X)
        preds = y_predicted_proba[:,1]
        fpr, tpr, threshold = metrics.roc_curve(y, preds)
        roc_auc = metrics.auc(fpr, tpr)
        
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    
    

class Gdbr():
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def fit(self):
        self.model = GradientBoostingClassifier(learning_rate=0.01,
                                  max_depth=10,
                                  n_estimators=1000,
                                  min_samples_leaf=100,
                                  max_features=4)
        self.model.fit(self.X, self.y)
        return self
    
    def params(self):
        print(self.model.get_params)
    
    def predict(self, X):
        y_predicted = self.model.predict(X)
        return y_predicted
    
    def predict_proba(self, X):
        y_proba = self.model.predict_proba(X)
        return y_proba
    
    def score(self, X, y):
        return self.model.score(X, y)

    def confusion_matrix_plot(self, y, y_predicted, cmap):
        cm = confusion_matrix(y, y_predicted)
        plt.clf()
        plt.imshow(cm, cmap, interpolation='nearest')
        classNames = ['Not Fraud','Fraud']
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        thresh = cm.max() / 2.
        s = [['TN','FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+"\n"+str(cm[i][j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
        plt.show()
        
    def confusion_matrix(self, y, y_predicted):
        return confusion_matrix(y, y_predicted).ravel()
    
    def feature_importances(self, X):
        top_cols = np.argsort(self.model.feature_importances_)
        importances =self.model.feature_importances_[top_cols]
        fig = plt.figure(figsize=(10, 10))
        x_ind = np.arange(importances.shape[0])
        plt.barh(x_ind, importances/importances[-1:], height=.3, align='center')
        plt.ylim(x_ind.min() -0.5, x_ind.max() + 0.5)
        plt.yticks(x_ind, X.columns[top_cols], fontsize=14)
        plt.show()
        
    def partial_dependence_plots(self, X):
        fig, axs = plot_partial_dependence(self.model, X, np.arange(X.shape[1]),
                    n_jobs=3, grid_resolution=100,feature_names = X.columns)
        fig.set_size_inches((20,30))
    
    def plot_roc(self, X, y):
        y_predicted_proba = self.model.predict_proba(X)
        preds = y_predicted_proba[:,1]
        fpr, tpr, threshold = metrics.roc_curve(y, preds)
        roc_auc = metrics.auc(fpr, tpr)
        
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


class RandomForest():
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def fit(self):
        self.model = RandomForestClassifier(n_estimators=1000,
                            n_jobs=-1)
        self.model.fit(self.X, self.y)
        return self
    
    def predict(self, X):
        y_predicted = self.model.predict(X)
        return y_predicted
    
    def predict_proba(self, X):
        y_proba = self.model.predict_proba(X)
        return y_proba
    
    def score(self, X, y):
        score = self.model.score(X, y)
        return score
    
    def confusion_matrix_plot(self, y, y_predicted,cmap):
        cm = confusion_matrix(y, y_predicted)
        plt.clf()
        plt.imshow(cm, cmap, interpolation='nearest')
        classNames = ['Not Fraud','Fraud']
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        thresh = cm.max() / 2.
        s = [['TN','FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+"\n"+str(cm[i][j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
        plt.show()
        
    def confusion_matrix(self, y, y_predicted):
        return confusion_matrix(y, y_predicted).ravel()
        
    def plot_roc(self, X, y):
        y_predicted_proba = self.model.predict_proba(X)
        preds = y_predicted_proba[:,1]
        fpr, tpr, threshold = metrics.roc_curve(y, preds)
        roc_auc = metrics.auc(fpr, tpr)
        
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()



class SupportVectorMachine():
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def scaling(self, X):
        cont_vars=['body_length', 'name_length', 'sale_duration', 'user_age',
        'org_facebook','org_twitter', 'avg_ticket_cost','tot_ticket_quant']
        X[cont_vars]=StandardScaler().fit_transform(X[cont_vars])
        return X
 
    def fit(self):
        self.model = SVC(gamma='scale',probability=True)
        X = self.scaling(self.X)
        self.model.fit(X, self.y)
        return self
    
    def predict(self, X):
        X = self.scaling(X)
        y_predicted = self.model.predict(X)
        return y_predicted
    
    def predict_proba(self, X):
        X = self.scaling(X)
        y_proba = self.model.predict_proba(X)
        return y_proba
    
    def score(self, X, y):
        X = self.scaling(X)
        score = self.model.score(X, y)
        return score
    
    def confusion_matrix_plot(self, y, y_predicted,cmap):
        cm = confusion_matrix(y, y_predicted)
        plt.clf()
        plt.imshow(cm, cmap, interpolation='nearest')
        classNames = ['Not Fraud','Fraud']
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        thresh = cm.max() / 2.
        s = [['TN','FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+"\n"+str(cm[i][j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
        plt.show()
        
    def confusion_matrix(self, y, y_predicted):
        return confusion_matrix(y, y_predicted).ravel()
            
    def plot_roc(self, X, y):
        X = self.scaling(X)
        y_predicted_proba = self.model.predict_proba(X)
        preds = y_predicted_proba[:,1]
        fpr, tpr, threshold = metrics.roc_curve(y, preds)
        roc_auc = metrics.auc(fpr, tpr)
        
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()



class NaiveBayes():
    def __init__(self,X,y):
        self.X = X
        self.y = y
            
    def vectorizer(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_vectorizer.fit(self.X)
        return self.tfidf_vectorizer        
    
    def fit(self):
        # Vectorize the text
        X_tr = self.vectorizer().transform(self.X)

        # Fit multinomial naive bayes model
        self.model = MultinomialNB()
        self.model.fit(X_tr, self.y)
        
        return self.model
    
    def predict(self,X):
        X_tr = self.vectorizer().transform(X)
        y_predicted = self.model.predict(X_tr)
        
        return y_predicted

    def score(self, X, y):
        X_tr = self.vectorizer().transform(X) 
        print(self.model.score(X_tr,y))
    
    def confusion_matrix(self, y, y_predicted):
        tn, fp, fn, tp=confusion_matrix(y, y_predicted).ravel()
        print('TN:',tn, 'FP:',fp, 'FN:',fn, 'TP:',tp)
    
    def plot_roc(self, X, y):
        X_tr = self.vectorizer().transform(X) 
        y_predicted_proba = self.model.predict_proba(X_tr)
        preds = y_predicted_proba[:,1]
        fpr, tpr, threshold = metrics.roc_curve(y, preds)
        roc_auc = metrics.auc(fpr, tpr)
        
        # method I: plt
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    

    
    

    
