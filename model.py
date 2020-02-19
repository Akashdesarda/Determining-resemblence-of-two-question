import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.layers import Dense, concatenate, Lambda, Activation, BatchNormalization, Input, Dropout
from keras.models import Model, Sequential
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, CSVLogger
# from keras.utils import to_categorical

def build(embedding_func,
          no_of_hidden_layer_1: int=512, 
          no_of_hidden_layer_2: int=512, 
          no_of_hidden_layer_3: int=512,
          dropout: float=0.25,
          verbosity: int=0):
    """Build network to determine resemblance of two question
    
    Parameters
    ----------
    embedding_func:
        Any function to generate word embedding
    no_of_hidden_layer_1 : int, optional
        No of input in hidden layer 1, by default 512
    no_of_hidden_layer_2 : int, optional
        No of input in hidden layer 2, by default 512
    no_of_hidden_layer_3 : int, optional
        No of input in hidden layer 2, by default 512
    dropout : float, optional
        Dropout value, by default 0.25
    verbosity : int, optional
        Print verbosity or model summary if > 0, by default 0
    """
    
    input1_layer = Input(shape=(1,), dtype=tf.string)
    embeding_input1_layer = Lambda(embedding_func, output_shape=(512,))(input1_layer)

    input2_layer = Input(shape=(1,), dtype=tf.string)
    embeding_input2_layer = Lambda(embedding_func, output_shape=(512,))(input2_layer)

    # Concatenating the both input layer
    merged = concatenate([embeding_input1_layer, embeding_input2_layer])

    #1 layer
    merged = Dense(no_of_hidden_layer_1, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(dropout)(merged)

    merged = Dense(no_of_hidden_layer_2, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(dropout)(merged)

    merged = Dense(no_of_hidden_layer_3, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(dropout)(merged)

    pred_layer = Dense(2, activation='softmax')(merged)
    model = Model(inputs=[input1_layer, input2_layer], outputs=pred_layer)
    if verbosity > 0:
        model.summary()
    return model