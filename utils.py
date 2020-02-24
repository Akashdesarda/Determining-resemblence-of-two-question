import warnings
warnings.filterwarnings('ignore')

import re
from typing import Dict
from matplotlib import pyplot as plt
from datetime import datetime
now = datetime.now().strftime("%d-%m-%Y")

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
    

