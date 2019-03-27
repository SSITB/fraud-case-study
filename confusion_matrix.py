import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Confusion matrix plot
# =============================================================================

def confusion_matrix_plot(cm, cmap):
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

