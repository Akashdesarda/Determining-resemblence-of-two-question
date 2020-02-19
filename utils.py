import warnings
warnings.filterwarnings('ignore')

from typing import Dict
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
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

def generate_embedding(pretrained_weights: str='./pretrained_weights/universal_sentence_encoder_large',
                       doc:str =None):
    """Any word embedding generator weights from https://tfhub.dev
    
    Parameters
    ----------
    pretrained_weights : str, optional
        path or url of pretrained weights, by default './pretrained_weights/universal_sentence_encoder_large'
    doc : str
        document to generate word embeddings
    """
    embed = hub.load(pretrained_weights)
    return embed(tf.squeeze(tf.cast(doc, tf.string)))

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
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
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
    

