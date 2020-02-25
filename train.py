import warnings
warnings.filterwarnings('ignore')

import numpy as np
np.random.seed(30)

from timeit import default_timer as timer
from datetime import datetime
now = datetime.now().strftime("%d-%m-%Y")

from sklearn.model_selection import train_test_split
import joblib

import tensorflow as tf
# import tensorflow_hub as hub
from keras import backend as K

from utils import clean_text, visualize
from similarity_net import SimilarityNet
from callbacks import callbacks
import mlflow

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
        
# Loading data
print('[INFO]...Loading Data')
x1 = np.load('./data/q1_use_embeddings_404287.npy')
x2 = np.load('./data/q1_use_embeddings_404287.npy')
labels = np.load('./data/labels.npy')
train_data = np.concatenate((x1, x2), axis=1)

sn = SimilarityNet()

# Building nn
model = sn.build_nn(verbosity=1)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("[INFO]...Model Build Completed")
# Preparing training & validation data

# Using the sklearn to split data in question1 and question2 train and test in the ration 80-20 %
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, labels, test_size=0.25, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.25, random_state=42) 
# Callbacks list
callbacks_list = callbacks()

#Mlflow config
mlflow_tag = {'SimilarityNet':'DenseIncremental', 'Activation':'sigmoid' ,'Dataset':'Quora question pair'}
mlflow.set_tracking_uri('http://127.0.0.1:5000')

with mlflow.start_run(experiment_id=2, run_name='using_embeddings', nested=True):
    mlflow.set_tags(mlflow_tag)
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        history = model.fit([x1_train, x2_train],
                            y_train,
                            validation_data=([x1_test, x2_test], y_test),
                            callbacks=callbacks_list,
                            epochs=20,
                            batch_size=512)

        visualize(history,
                save_dir=f'./assets/logs/history-{now}.png')
        
# Training on Random Forest Classifier
rf_model = sn.build_RandomForestClassifier()
rf_model.fit(x_train, y_train)
joblib.dump(rf_model, './assets/RandomForest/SmilarityNet-RandomForest-nest500.pkl')

# Training on CatBoost Classifier
cb_model = sn.build_CatBoostClassifier()
cb_model.fit(x_train, y_train)
cb_model.save_model('./assets/weights/Catboost/SimilarityNet-Catboost-depth10')