import warnings
warnings.filterwarnings('ignore')

import numpy as np
from keras.models import load_model
import tensorflow_hub as hub
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
model = hub.load(module_url)
print (f"module {module_url} loaded")
def embed(input):
  return model(input)

# Taking question as Input
q1 = input("Type Question 1 here -->")
q2 = input("Type Question 2 here -->")

que = [q1, q2]

# Generating embedding 
inq1 = np.array(embed([que[0]]))
inq2 = np.array(embed([que[1]]))

tf.compat.v1.disable_eager_execution()

# Loading the save weights
model = load_model('assets/weights/DenseIncrementalSigmoid/exp2/SimilarityNet-epoch:12-val_acc:0.78.hdf5')
                    
# Predicting the similarity between the two input questions 
predicts = model.predict([inq1, inq2])
predict_logits = predicts.argmax(axis=1)
# print("----FINAL RESULT----")
# if(predict_logits[0] == 1):
#     print("****Questions are Similar****")
# else:
#     print("****Questions are not Similar****")

print(predict_logits)
print(predicts)
#TODO: 1. Complete Inference logic & its validation. 