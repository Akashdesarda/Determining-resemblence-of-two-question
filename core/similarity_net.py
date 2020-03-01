
import warnings
warnings.filterwarnings('ignore')

from keras.layers import Dense, concatenate, Activation, BatchNormalization, Dropout
from keras.models import Model, Input, Sequential
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


class SimilarityNet:
    """Build backend for Classification Job. Currently three backend are supported:
       1. Random Forest Classifier
       2. Catboost classifier
       3. NeuralNet
    """
    # def __init__(self, train_data, train_label):
    #     """Build a classification model
        
    #     Parameters
    #     ----------
    #     train_data : np array or pandas series
    #         training data to fit over model
    #     train_label : np array or pandas series
    #         training label
        # """
    
    @staticmethod
    def build_RandomForestClassifier(bootstrap:bool=False, criterion:str='gini', 
                                     max_depth:int=None, min_samples_split:int=2,
                                     n_estimators=500,n_jobs:int=-1, verbose=0):
        """Build a Random forest classifier. Default parameter currently gives best result
        
        Parameters
        ----------
        bootstrap : bool, optional
            enable bootstrap, by default False
        criterion : str, optional
            loss function criteria. 'entropy','gini' are supported, by default 'gini'
        max_depth : int, optional
            max depth to split tree, by default None
        min_samples_split : int, optional
            min sample needed to split, by default 2
        n_estimators : int, optional
            no of tree to grow, by default 500
        n_jobs : int, optional
            to run jobs on all processor parallely,-1: to run on all processor & 0: to run on single, by default -1
        verbose : int, optional
            print details durinf training, by default 0
        """
        model = RandomForestClassifier(bootstrap=bootstrap, criterion=criterion, 
                                     max_depth=max_depth, min_samples_split=min_samples_split,
                                     n_estimators=n_estimators, n_jobs=n_jobs, verbose=verbose)
        return model
    
    @staticmethod
    def build_CatBoostClassifier(loss_function:str='Logloss', learning_rate:float=0.1,
                                 l2_leaf_reg:int=7,depth:int=10, use_gpu:bool=True):
        """Build a CatBoost Classifier. Default parameter gives best result
        
        Parameters
        ----------
        loss_function : str, optional
            loss to be used, by default 'Logloss'
        learning_rate : float, optional
            learning rate, by default 0.1
        l2_leaf_reg : int, optional
            l2 regularization rate at leaf level, by default 7
        depth : int, optional
            depth to grow a tree, by default 10
        use_gpu : bool, optional
            use cuda enabled gpu acceleration, by default True
        """
        if use_gpu is not False:
            model = CatBoostClassifier(task_type="GPU",devices='0:1', loss_function=loss_function, 
                                       learning_rate=learning_rate,l2_leaf_reg=l2_leaf_reg,depth=depth)
        else:
            model = CatBoostClassifier(loss_function=loss_function, learning_rate=learning_rate,
                                       l2_leaf_reg=l2_leaf_reg,depth=depth)
            
        return model
    
    @staticmethod
    def build_nn(dropout: float=0.3,verbosity: int=0):
        """Build network to determine resemblance of two question
        
        Parameters
        ----------
        dropout : float, optional
            Dropout value, by default 0.25
        verbosity : int, optional
            Print verbosity or model summary if > 0, by default 0
        """
        model = Sequential()
        model.add(Dense(1024, input_shape=(1024,), activation='relu', kernel_regularizer=regularizers.l2(0.02)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(Dense(1, activation='sigmoid'))
        
        if verbosity > 0:
            model.summary()
        return model
    
    @staticmethod
    def build_nn_experimental(dropout: float=0.3, verbosity: int=0):
        """Build network to determine resemblance of two question using Keras Functional API
           Note: This is experimental Neural net where embeddings of both question are feed to network seprately. So there will be two input.
        
        Parameters
        ----------
        dropout : float, optional
            Dropout value, by default 0.25
        verbosity : int, optional
            Print verbosity or model summary if > 0, by default 0
        """
        # Setting Up Input layer
        input_q1 = Input(shape=(512,))
        input_q2 = Input(shape=(512,))
        
        # Network for 1st input Dense 128 --> Relu --> Dense 264 --> Relu
        input1_layer = Dense(512, activation='relu')(input_q1)
        input1_layer = BatchNormalization()(input1_layer)
        input1_layer = Dropout(dropout)(input1_layer)
        
        input1_layer = Dense(512, activation='relu')(input1_layer)
        input1_layer = BatchNormalization()(input1_layer)
        input1_layer = Dropout(dropout)(input1_layer)
        
        input1_layer = Model(inputs=input_q1, outputs=input1_layer)
        
        # Network for 2st input Dense 128 --> Relu --> Dense 264 --> Relu
        input2_layer = Dense(512, activation='relu')(input_q2)
        input2_layer = BatchNormalization()(input2_layer)
        input2_layer = Dropout(dropout)(input2_layer)
        
        input2_layer = Dense(512, activation='relu')(input2_layer)
        input2_layer = BatchNormalization()(input2_layer)
        input2_layer = Dropout(dropout)(input2_layer)
        
        input2_layer = Model(inputs=input_q2, outputs=input2_layer)
        
        merged = concatenate([input1_layer.output, input2_layer.output])

        # Fully connected layer & final prediction layer
        pred_layer = Dense(4096, activation='relu')(merged)
        pred_layer = Dense(1024, activation='relu')(pred_layer)
        pred_layer = Dense(256, activation='relu')(pred_layer)
        pred_layer = Dense(64, activation='relu')(pred_layer)
        pred_layer = Dropout(dropout)(pred_layer)
        
        pred_layer = Dense(1, activation='sigmoid')(pred_layer)
        
        model = Model(inputs=[input1_layer.input, input2_layer.input], outputs=pred_layer)
        if verbosity > 0:
            model.summary()
        return model
                                                                                                                                                                                