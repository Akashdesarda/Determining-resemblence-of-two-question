from typing import Dict, List

import numpy as np
import tensorflow as tf
from core.embeddings_generator import Generator

from keras.models import load_model
from catboost import CatBoostClassifier
import joblib

tf.compat.v1.enable_eager_execution()

def predict(config: Dict):
    """Predict Similarity on given two question

    Parameters
    ----------
    config : Dict
        Config yaml/json containing respective value
    """
    # Generating embedding 
    generator = Generator(config['embedding']['model_url'])
    if config['embedding']['job'] == 'unit':
        que = config['inference']['input']
        inq1 = np.array(generator.unit_generator([que[0]]))
        inq2 = np.array(generator.unit_generator([que[1]]))
        inque = np.concatenate(inq1, inq2, axis=1)
    
    tf.compat.v1.disable_eager_execution()
    
    if config['inference']['backend_classifier'] == 'catboost':
        model = CatBoostClassifier()
        model.load_model(config['inference']['model_path'])
        result = model.predict(inque)
        
    elif config['inference']['backend_classifier'] == 'random forest':
        model = joblib.load(config['inference']['model_path'])
        result = model.predict(inque)
    
    elif config['inference']['backend_classifier'] == 'neural net':
        model = load_model(config['inference']['model_path'])
        result_proba = model.predict(inque)
        result = result_proba.argmax(axis = 1)
        
    print("[INFO]...Results are:")
    if(result[0] == 1):
        print("****Questions are Similar****")
    else:
        print("****Questions are not Similar****")