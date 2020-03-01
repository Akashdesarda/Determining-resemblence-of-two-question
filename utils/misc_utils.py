import warnings
warnings.filterwarnings('ignore')

import re
from typing import Dict
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
now = datetime.now().strftime("%d-%m-%Y")

def limit_gpu():
    # Tf GPU memory graph
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print('[INFO]... ',len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def clean_text(doc):
    """Cleaning of text data
    
    Parameters
    ----------
    doc : str
        Any document
    
    Returns
    -------
    str
        cleaned document
    """
    if type(doc) == str:
        #lowercasing the text
        text = doc.lower()
        # Removing non ASCII chars
        text = re.sub(r'[^\x00-\x7f]',r' ',text)
        return text
    else:
        return ""

def visualize(history: Dict, save_plot: bool=True, save_dir: str=None):
    """Visualize training history of model
    
    Parameters
    ----------
    history : Dict
        model.fit history
    save_plot : bool, optional
        save plot to hard disk, by default True
    save_dir : str, optional
        path to save plot, by default None
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    if save_plot is not False:
        plt.savefig(save_dir)
    else:
        plt.show()
    
def report(y_true, y_pred):
    """Logging of report
    
    Parameters
    ----------
    y_true : numpy array or pandas series
        lables of test data
    y_pred : numpy array or pandas series
        labels of predicted data
    """
    print(f"[REPORT]...Accuracy of recently trained model is: {accuracy_score(y_true, y_pred)}")
    print("Following is detailed classification Report")
    print(classification_report(y_true, y_pred))
