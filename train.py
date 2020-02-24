import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
np.random.seed(30)

from timeit import default_timer as timer
from datetime import datetime
now = datetime.now().strftime("%d-%m-%Y")

from sklearn.model_selection import train_test_split

import tensorflow as tf
# import tensorflow_hub as hub
from keras import backend as K
from keras.utils import to_categorical

from utils import clean_text, visualize
from SimilarityNet import build
from callbacks import callbacks
import mlflow

data = pd.read_csv('./data/train_que.csv')
data.dropna(inplace=True)
# print("[INFO]...Data loaded Successfully")
# data.dropna(inplace = True)
# data['question1'], data['question2'] = data['question1'].apply(clean_text), data['question2'].apply(clean_text)
# print("[INFO]...Data transformation completed")

# #Loadind universal sentence encoder
# print("[INFO]...Loadind universal sentence encoder")
# embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')
# # creating a method for embedding and will using method for every input layer 
# def UniversalEmbedding(doc):
#     return embed(tf.squeeze(tf.cast(doc, tf.string)))
# print("[INFO]...Pretrained weights from tensorflow hub successfully loaded")


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
        
# Building network
model = build(verbosity=1)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("[INFO]...Model Build Completed")
# Preparing training & validation data
x1 = np.load('./data/q1_use_embeddings_404287.npy')
x2 = np.load('./data/q1_use_embeddings_404287.npy')
labels = pd.get_dummies(data['is_duplicate']) 
# Using the sklearn to split data in question1 and question2 train and test in the ration 80-20 %
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, labels, test_size=0.25, random_state=42)

# train_q1 = x1_train.tolist()
# train_q1 = np.array(train_q1, dtype=object)[:, np.newaxis]
# train_q2 = x2_train.tolist()
# train_q2 = np.array(train_q2, dtype=object)[:, np.newaxis]

# #train_labels = np.asarray(pd.get_dummies(y_train), dtype = np.int8)
# train_labels = to_categorical(y_train, num_classes=2)

# test_q1 = x1_test.tolist()
# test_q1 = np.array(test_q1, dtype=object)[:, np.newaxis]
# test_q2 = x2_test.tolist()
# test_q2 = np.array(test_q2, dtype=object)[:, np.newaxis]

# test_labels = to_categorical(y_test, num_classes=2)

# Callbacks list
callbacks_list = callbacks()

#Mlflow config
mlflow_tag = {'SimilarityNet':'DenseIncremental', 'Activation':'Sigmoid' ,'Dataset':'Quora question pair'}
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
        
        # json_config = model.to_json()
        # with open('model_config.json', 'w') as json_file:
        #     json_file.write(json_config)