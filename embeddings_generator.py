import tensorflow_hub as hub
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

#Incomplete
# #Loadind universal sentence encoder
# print("[INFO]...Loadind universal sentence encoder")
# embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')
# # creating a method for embedding and will using method for every input layer 
# def UniversalEmbedding(doc):
#     return embed(tf.squeeze(tf.cast(doc, tf.string)))
# print("[INFO]...Pretrained weights from tensorflow hub successfully loaded")