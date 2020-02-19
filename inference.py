import numpy as np
import keras.models
q1 = input("Type Question 1 here -->")
q2 = input("Type Question 2 here -->") 
q1 = np.array([[q1],[q1]])
q2 = np.array([[q2],[q2]])



# Loading the save weights
model = keras.models.model_from_json()
model.load_weights()
 
# Predicting the similarity between the two input questions 
predicts = model.predict([q1, q2], verbose=0)
predict_logits = predicts.argmax(axis=1)
print("----FINAL RESULT----")
if(predict_logits[0] == 1):
    print("****Questions are Similar****")
else:
    print("****Questions are not Similar****")

#TODO: 1. Complete Inference logic & its validation. 