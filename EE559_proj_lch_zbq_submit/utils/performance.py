import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_cm(label_true, label_pred, file_name):
    cm = confusion_matrix(label_true, label_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure()
    disp.plot()
    # plt.show()
    plt.savefig(file_name)








