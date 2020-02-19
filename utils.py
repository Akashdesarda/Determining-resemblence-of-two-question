import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_hub as hub

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
    

