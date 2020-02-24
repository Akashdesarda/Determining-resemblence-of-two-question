import warnings
warnings.filterwarnings('ignore')

from keras.layers import Dense, concatenate, Activation, BatchNormalization, Dropout
from keras.models import Model, Input
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, CSVLogger
# from keras.utils import to_categorical

def build(no_of_hidden_layer_1: int=128, 
          no_of_hidden_layer_2: int=264, 
          no_of_hidden_layer_3: int=512,
          dropout: float=0.25,
          verbosity: int=0):
    """Build network to determine resemblance of two question using Keras Functional API
    
    Parameters
    ----------
    embedding_func:
        Any function to generate word embedding
    no_of_hidden_layer_1 : int, optional
        No of input in hidden layer 1, by default 128
    no_of_hidden_layer_2 : int, optional
        No of input in hidden layer 2, by default 264
    no_of_hidden_layer_3 : int, optional
        No of input in hidden layer 2, by default 512
    dropout : float, optional
        Dropout value, by default 0.25
    verbosity : int, optional
        Print verbosity or model summary if > 0, by default 0
    """
    # Setting Up Input layer
    input_q1 = Input(shape=(512,))
    input_q2 = Input(shape=(512,))
    
    # Network for 1st input Dense 128 --> Relu --> Dense 264 --> Relu
    input1_layer = Dense(128, activation='relu')(input_q1)
    input1_layer = BatchNormalization()(input1_layer)
    input1_layer = Dropout(0.3)(input1_layer)
    
    input1_layer = Dense(128, activation='relu')(input1_layer)
    input1_layer = BatchNormalization()(input1_layer)
    input1_layer = Dropout(0.3)(input1_layer)
    
    input1_layer = Dense(264, activation='relu')(input1_layer)
    input1_layer = BatchNormalization()(input1_layer)
    input1_layer = Dropout(0.3)(input1_layer)
    
    input1_layer = Dense(264, activation='relu')(input1_layer)
    input1_layer = BatchNormalization()(input1_layer)
    input1_layer = Dropout(0.3)(input1_layer)
    
    input1_layer = Model(inputs=input_q1, outputs=input1_layer)
    
    # Network for 2st input Dense 128 --> Relu --> Dense 264 --> Relu
    input2_layer = Dense(128, activation='relu')(input_q2)
    input2_layer = BatchNormalization()(input2_layer)
    input2_layer = Dropout(0.3)(input2_layer)
    
    input2_layer = Dense(128, activation='relu')(input2_layer)
    input2_layer = BatchNormalization()(input2_layer)
    input2_layer = Dropout(0.3)(input2_layer)
    
    input2_layer = Dense(264, activation='relu')(input2_layer)
    input2_layer = BatchNormalization()(input2_layer)
    input2_layer = Dropout(0.3)(input2_layer)
    
    input2_layer = Dense(264, activation='relu')(input2_layer)
    input2_layer = BatchNormalization()(input2_layer)
    input2_layer = Dropout(0.3)(input2_layer)
    
    input2_layer = Model(inputs=input_q2, outputs=input2_layer)
    
    # input1_layer = Input(shape=(512,))
    # # embeding_input1_layer = Lambda(embedding_func, output_shape=(512,))(input1_layer)

    # input2_layer = Input(shape=(512,))
    # # embeding_input2_layer = Lambda(embedding_func, output_shape=(512,))(input2_layer)

    # Concatenating the both input layer
    merged = concatenate([input1_layer.output, input2_layer.output])

    # #1 layer
    # merged = Dense(no_of_hidden_layer_1, activation='relu')(merged)
    # merged = BatchNormalization()(merged)
    # merged = Dropout(dropout)(merged)

    # merged = Dense(no_of_hidden_layer_2, activation='relu')(merged)
    # merged = BatchNormalization()(merged)
    # merged = Dropout(dropout)(merged)

    # merged = Dense(no_of_hidden_layer_3, activation='relu')(merged)
    # merged = BatchNormalization()(merged)
    # merged = Dropout(dropout)(merged)
    # Fully connected layer & final prediction layer
    pred_layer = Dense(512, activation='relu')(merged)
    pred_layer = BatchNormalization()(pred_layer)
    pred_layer = Dropout(0.3)(pred_layer)
    
    pred_layer = Dense(512, activation='relu')(pred_layer)
    pred_layer = BatchNormalization()(pred_layer)
    pred_layer = Dropout(0.2)(pred_layer)
    
    pred_layer = Dense(2, activation='softmax')(pred_layer)
    
    model = Model(inputs=[input1_layer.input, input2_layer.input], outputs=pred_layer)
    if verbosity > 0:
        model.summary()
    return model