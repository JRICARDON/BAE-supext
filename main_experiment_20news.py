import numpy as np
import keras,gc,nltk
import pandas as pd
nltk.download('reuters')
nltk.download('wordnet')

from load_20news import *
from load_snippets import *
from load_reuters import *

from text_representation import *

print("\n=====> Loading data ...\n")

texts_t, labels_t, texts_test, labels_test, list_dataset_labels = load_20news()#load_snippets()#

print(list_dataset_labels)

from sklearn.model_selection import train_test_split
labels_t = np.asarray(labels_t)
labels_test = np.asarray(labels_test)
texts_train,texts_val,labels_train,labels_val  = train_test_split(texts_t,labels_t,random_state=20,test_size=0.25)

print("Size Training Set: ",len(texts_train))
print("Size Validation Set: ",len(texts_val))
print("Size Test Set: ",len(texts_test))

print("\n=====> Vectorizing Text ...\n")

vectors_train, vectors_val, vectors_test = represent_text(texts_train,texts_val,texts_test,model='TF')

print(type(vectors_train))
X_train = np.asarray(vectors_train.todense())
X_val = np.asarray(vectors_val.todense())
X_test = np.asarray(vectors_test.todense())

del vectors_train,vectors_val,vectors_test
gc.collect()

print("\n=====> Smoothing TF  ...\n")

##soft TF - much better!
X_train_input = np.log(X_train+1.0) 
X_val_input = np.log(X_val+1.0) 
X_test_input = np.log(X_test+1.0) 

print(X_train_input.shape,X_val_input.shape,X_test_input.shape)

print("\n=====> Creating and Training the Models (VDSH and BAE) ... \n")

from models import *

batch_size = 100

X_total_input = np.concatenate((X_train_input,X_val_input),axis=0)
X_total = np.concatenate((X_train,X_val),axis=0)
labels_total = np.concatenate((labels_train,labels_val),axis=0)

traditional_vae,encoder_Tvae,generator_Tvae = traditional_VAE(X_train.shape[1],Nb=32,units=500,layers_e=2,layers_d=0)
traditional_vae.fit(X_total_input, X_total, epochs=50, batch_size=batch_size,verbose=0)

binary_vae,encoder_Bvae,generator_Bvae = binary_VAE(X_train.shape[1],Nb=32,units=500,layers_e=2,layers_d=2)
binary_vae.fit(X_total_input, X_total, epochs=50, batch_size=batch_size,verbose=0)

print("\n=====> Evaluate the Models using KNN Search ... \n")

from similarity_search import *

dataset = "20news"
k_topk = 100

p_t,r_t = evaluate_hashing(list_dataset_labels, encoder_Tvae,X_total_input,X_test_input,labels_total,labels_test,traditional=True,tipo="topK")
p_b,r_b = evaluate_hashing(list_dataset_labels, encoder_Bvae,X_total_input,X_test_input,labels_total,labels_test,traditional=False,tipo="topK")

file = open("Results_Top_K_20news.csv","w")
file.write("20news, VDSH, %d, %f, %f\n"%(k_topk,p_t,r_t))
file.write("20news, BAE, %d, %f, %f\n"%(k_topk,p_b,r_b))
file.close()

print("DONE ...")

print("\n=====> Evaluate the Models using Range/Ball Search ... \n")

ball_radius = np.arange(0,10) #ball of radius graphic

binary_p = []
binary_r = []
encode_total = encoder_Bvae.predict(X_total_input)
encode_test = encoder_Bvae.predict(X_test_input)
probas_total= keras.activations.sigmoid(encode_total).eval(session=K.get_session())
probas_test= keras.activations.sigmoid(encode_test).eval(session=K.get_session())
total_hash_b = (probas_total > 0.5)*1
test_hash_b = (probas_test > 0.5)*1
    
traditional_p = []
traditional_r = []
encode_total = encoder_Tvae.predict(X_total_input)
encode_test = encoder_Tvae.predict(X_test_input)
median= MedianHashing()
median.fit(encode_total)
total_hash_t = median.transform(encode_total)
test_hash_t = median.transform(encode_test)

file2 = open("Results_BallSearch_20news.csv","w")

for ball_r in ball_radius:

	test_similares_train =  get_similar(test_hash_b,total_hash_b,tipo='ball',ball=ball_r) 
	p_b,r_b  = measure_metrics(list_dataset_labels,test_similares_train,labels_test,labels_destination=labels_total)

	test_similares_train =  get_similar(test_hash_t,total_hash_t,tipo='ball',ball=ball_r)
	p_t,r_t  = measure_metrics(list_dataset_labels,test_similares_train,labels_test,labels_destination=labels_total)
	
	file2.write("20news, VDSH, %d, %f, %f\n"%(ball_r,p_t,r_t))
	file2.write("20news, BAE, %d, %f, %f\n"%(ball_r,p_b,r_b))

file2.close()
print("DONE ... ")