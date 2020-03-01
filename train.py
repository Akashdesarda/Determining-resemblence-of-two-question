import warnings
warnings.filterwarnings('ignore')

import numpy as np
np.random.seed(30)

from datetime import datetime
now = datetime.now().strftime("%d-%m-%Y")

from sklearn.model_selection import train_test_split
import joblib

import tensorflow as tf

from utils.misc_utils import visualize, report
from core.similarity_net import SimilarityNet
from utils.callbacks import callbacks

def train(config: Dict):
    # Loading data
    print('[INFO]...Loading Data')
    x1 = np.load('./data/q1_use_embeddings_404287.npy')
    x2 = np.load('./data/q1_use_embeddings_404287.npy')
    labels = np.load('./data/labels.npy')
    train_data = np.concatenate((x1, x2), axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.25)
    
    sn = SimilarityNet()
    
    if config['training']['backend'] == 'random forest':
        model = sn.build_RandomForestClassifier()
        model.fit(x_train, y_train)
        joblib.dump(model, config['training']['model_save_path'])
    
    elif config['training']['backend'] == 'catboost':
        model = sn.build_CatBoostClassifier()
        model.fit(x_train, y_train)
        model.save_model(config['training']['model_save_path'])
    
    elif config['training']['backend'] == 'neural net':
        callbacks_list = callbacks(save_path=config['training']['model_save_path'])
        model = sn.build_nn()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(x_train,
                            y_train,
                            validation_data=(x_test, y_test),
                            callbacks=callbacks_list,
                            epochs=config['training']['neural_net']['epoch'],
                            batch_size=config['training']['neural_net']['batch_size'])

        visualize(history,
                save_dir=f'./assets/logs/history-{now}.png')
        
    if config['training']['log_report']:
        if config['training']['backend'] == 'random forest' or 'catboost':
            y_pred = model.predict(x_test)
        if config['training']['backend'] == 'neural net':
            y_pred = model.predict_classes(x_test)
        report(y_test, y_pred)
    