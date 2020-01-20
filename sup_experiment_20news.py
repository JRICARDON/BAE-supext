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

dataset_name = "20news"
texts_t, labels_t, texts_test, labels_test, list_dataset_labels = load_20news()

unsup_model = 3#choose between 1,2,3,4,5
### 1: sBAE1: atempts to predict the class label from the representation z learnt by the hidden layer immediately after the encoder's output layer, i.e. the representation learnt by the encoder is used to produce the hash AND to predict the class distribution.
###    training loss = unsupervised BAE loss + cross_entropy(y,pred_y_from_encoder)

### 2: sBAE2: identical to sBAE1 but implemented using different tools of the library (keras). 
###    training loss = unsupervised BAE loss + cross_entropy(y,pred_y_from_encoder)

### 3: sBAE3: atempts to predict the class label from the representation z' learnt by the hidden layer immediately after the decoder's output layer, i.e., the decoder should be able to reconstruct both the input x and its class y from the binary hash code.
###    training loss = unsupervised BAE loss + cross_entropy(y,pred_y_from_decoder)

### 4: sBAE4: atempts to predict the class label from the representation z' learnt by the hidden layer immediately after the decoder's output layer, i.e., the decoder should be able to reconstruct both the input x and its class y from the binary hash code.
###    training loss = unsupervised BAE loss + hamming_loss
###    hamming_loss = sum_{similar pairs x,x'} dist(code(x),code(x')) - sum_{dissimilar pairs x,x'} dist(code(x),code(x'))

### 5: sBAE5: combines sBAE3 and sBAE4
###    training loss = unsupervised BAE loss + hamming_loss + cross_entropy(y,pred_y_from_decoder)

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

print("\n=====> Encoding Labels ...\n")

from keras.utils import to_categorical
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(list_dataset_labels)

n_classes = len(list_dataset_labels)

y_train = label_encoder.transform(labels_train)
y_val = label_encoder.transform(labels_val)
y_test = label_encoder.transform(labels_test)

y_train_input = to_categorical(y_train,num_classes=n_classes)
y_val_input = to_categorical(y_val,num_classes=n_classes)
y_test_input = to_categorical(y_test,num_classes=n_classes)

print(y_train_input.shape, y_val_input.shape, y_test_input.shape)

print("\n=====> Creating and Training the Models (VDSH and BAE) ... \n")

from supervised_models import *

batch_size = 100

X_total_input = np.concatenate((X_train_input,X_val_input),axis=0)
X_total = np.concatenate((X_train,X_val),axis=0)

Y_total_input = np.concatenate((y_train_input,y_val_input),axis=0)

labels_total = np.concatenate((labels_train,labels_val),axis=0)

traditional_vae,encoder_Tvae,generator_Tvae = traditional_VAE(X_train.shape[1],n_classes,Nb=32,units=500,layers_e=2,layers_d=0)
traditional_vae.fit(X_total_input, [X_total, Y_total_input], epochs=50, batch_size=batch_size,verbose=1)

if unsup_model == 1:#choose between 1,2,3,4,5

	binary_vae,encoder_Bvae,generator_Bvae = sBAE1(X_train.shape[1],n_classes,Nb=32,units=500,layers_e=2,layers_d=2)
	binary_vae.fit([X_total_input,Y_total_input], X_total, epochs=50, batch_size=batch_size,verbose=1)
	name_model = 'sBAE1'

elif unsup_model == 2:

	binary_vae,encoder_Bvae,generator_Bvae = sBAE2(X_train.shape[1],n_classes,Nb=32,units=500,layers_e=2,layers_d=2)
	binary_vae.fit(X_total_input, [X_total, Y_total_input], epochs=50, batch_size=batch_size,verbose=1)
	name_model = 'sBAE2'

elif unsup_model == 3:

	binary_vae,encoder_Bvae,generator_Bvae = sBAE3(X_train.shape[1],n_classes,Nb=32,units=500,layers_e=2,layers_d=2)
	binary_vae.fit(X_total_input, [X_total, Y_total_input], epochs=50, batch_size=batch_size,verbose=1)
	name_model = 'sBAE3'

elif unsup_model == 4:

	binary_vae,encoder_Bvae,generator_Bvae = sBAE4(X_train.shape[1],n_classes,Nb=32,units=500,layers_e=2,layers_d=2)
	binary_vae.fit([X_total_input,Y_total_input], X_total, epochs=50, batch_size=batch_size,verbose=1)
	name_model = 'sBAE4'

else: #elif unsup_model == 5

	binary_vae,encoder_Bvae,generator_Bvae = sBAE5(X_train.shape[1],n_classes,Nb=32,units=500,layers_e=2,layers_d=2)
	binary_vae.fit([X_total_input,Y_total_input], X_total, epochs=50, batch_size=batch_size,verbose=1)
	name_model = 'sBAE5'

print("\n=====> Evaluate the Models using KNN Search ... \n")

from similarity_search import *

k_topk = 100

p_t,r_t = evaluate_hashing(list_dataset_labels, encoder_Tvae,X_total_input,X_test_input,labels_total,labels_test,traditional=True,tipo="topK")
p_b,r_b = evaluate_hashing(list_dataset_labels, encoder_Bvae,X_total_input,X_test_input,labels_total,labels_test,traditional=False,tipo="topK")

file = open("SUP_Results_Top_K_%s.csv"%dataset_name,"a")
file.write("%s, sVDSH, %d, %f, %f\n"%(dataset_name,k_topk,p_t,r_t))
file.write("%s, %s, %d, %f, %f\n"%(dataset_name,name_model,k_topk,p_b,r_b))
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

file2 = open("SUP_Results_BallSearch_%s.csv"%dataset_name,"a")

for ball_r in ball_radius:

	test_similares_train =  get_similar(test_hash_b,total_hash_b,tipo='ball',ball=ball_r) 
	p_b,r_b  = measure_metrics(list_dataset_labels,test_similares_train,labels_test,labels_destination=labels_total)

	test_similares_train =  get_similar(test_hash_t,total_hash_t,tipo='ball',ball=ball_r)
	p_t,r_t  = measure_metrics(list_dataset_labels,test_similares_train,labels_test,labels_destination=labels_total)
	
	file2.write("%s, sVDSH, %d, %f, %f\n"%(dataset_name,ball_r,p_t,r_t))
	file2.write("%s, %s, %d, %f, %f\n"%(dataset_name,name_model,ball_r,p_b,r_b))

file2.close()
print("DONE ... ")


