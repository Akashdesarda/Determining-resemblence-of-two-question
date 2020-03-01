import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from typing import Dict, List
from tqdm import tqdm

if tf.__version__[0] == '1': 
    tf.compat.v1.enable_eager_execution()

class Generator:
    """Generate text embedding
    """
    def __init__(self, model_url: str):
        """Generate text embeddings using any desired tensorflow_hub module
        
        Parameters
        ----------
        model_url : str
            Path of tensorflow_hub module to be used. It can be either a locally downloaded path or a direct url
        """
        self.model_url = model_url
        # self.model_url = config['embedding']['model_url']
        if 'https' in model_url:
            print("[WARNING]...You are using URL to load embedding generator backend & will take more time to load.\nThe module will be downloaded first (if not already) & then loaded. This is even true after every reboot of system. ")
        print("[INFO]...Loading embedding generator backend")
        self.model = hub.load(model_url)
        print (f"[INFO]...Module {model_url} loaded")
        
    def unit_generator(self, input):
        """Generate text embedding for single or unit example 
        
        Parameters
        ----------
        input : str
            A single doc to extract text embedding
        """
        return self.model(input)
    
    def batch_generator(self, docs: List):
        """Generate text embeddings for batches of docs
        Note: Here a doc can be a word, sentence or even a small paragraph
        
        Parameters
        ----------
        docs : List
            list of all docs or sentences
        
        Returns
        -------
        numpy.ndarray
            numpy array with shape of (n, 512), where n = len(docs)
        """
        print("[INFO]...Extracting embeddings on batch of {len(docs)}")
        embd_array = np.array([])
        embd_array = embd_array.reshape(-1, 512)
        for doc in tqdm(docs):
            _unit_embd = Generator.unit_generator([doc])
            _unit_embd_array = np.asarray(_unit_embd)
            embd_array = np.append(embd_array,_unit_embd_array, axis=0)
        return embd_array
    
    