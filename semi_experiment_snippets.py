import sys
import numpy as np
import keras, gc, nltk
import pandas as pd


batch_size = 100
epochs = 50
multilabel = False
max_radius = 30


nltk.download('reuters')
nltk.download('wordnet')

from load_20news import *
from load_snippets import *
from load_reuters import *

from text_representation import *

print("\n=====> Loading data ...\n")

dataset_name = "snippets"
texts_t, labels_t, texts_test, labels_test, list_dataset_labels = load_snippets()

print(list_dataset_labels)

from sklearn.model_selection import train_test_split

labels_t = np.asarray(labels_t)
labels_test = np.asarray(labels_test)

test_size = int(sys.argv[1])
test_size = test_size/100
texts_train, texts_val, labels_train, labels_val = train_test_split(texts_t, labels_t, random_state=20, test_size=test_size)

print("Size Training Set: ", len(texts_train))
print("Size Validation Set: ", len(texts_val))
print("Size Test Set: ", len(texts_test))

print("\n=====> Vectorizing Text ...\n")

vectors_train, vectors_val, vectors_test = represent_text(texts_train, texts_val, texts_test, model='TF')

print(type(vectors_train))
X_train = np.asarray(vectors_train.todense())
X_val = np.asarray(vectors_val.todense())
X_test = np.asarray(vectors_test.todense())

del vectors_train, vectors_val, vectors_test
gc.collect()

print("\n=====> Smoothing TF  ...\n")

##soft TF - much better!
X_train_input = np.log(X_train + 1.0)
X_val_input = np.log(X_val + 1.0)
X_test_input = np.log(X_test + 1.0)

print(X_train_input.shape, X_val_input.shape, X_test_input.shape)

print("\n=====> Encoding Labels ...\n")

from keras.utils import to_categorical
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(list_dataset_labels)

n_classes = len(list_dataset_labels)

y_train = label_encoder.transform(labels_train)
y_val = label_encoder.transform(labels_val)
y_test = label_encoder.transform(labels_test)

y_train_input = to_categorical(y_train, num_classes=n_classes)
y_val_input = to_categorical(y_val, num_classes=n_classes)

#y_val_input = to_categorical(y_val, num_classes=n_classes)
y_zeros = [0 for i in range(y_val_input.shape[1])]
y_val_input_new = np.array([y_zeros for i in range(y_val_input.shape[0])])
y_val_input = y_val_input_new

y_test_input = to_categorical(y_test, num_classes=n_classes)

print(y_train_input.shape, y_val_input.shape, y_test_input.shape)

print("\n=====> Creating and Training the Models (VDSH and BAE) ... \n")

from semisupervised_models import *

batch_size = 100

X_total_input = np.concatenate((X_train_input, X_val_input), axis=0)
X_total = np.concatenate((X_train, X_val), axis=0)

Y_total_input = np.concatenate((y_train_input, y_val_input), axis=0)
labels_total = np.concatenate((labels_train, labels_val), axis=0)

#################### -------Semi-Supervised BAE ---------- ######################


binary_vae, encoder_Bvae, generator_Bvae = sBAE3(X_train.shape[1], n_classes, Nb=32, units=500, layers_e=2, layers_d=2)
binary_vae.fit(X_total_input, [X_total, Y_total_input], epochs=50, batch_size=batch_size, verbose=2)
name_model = 'semi_BAE'


traditional_vae, encoder_Tvae, generator_Tvae = traditional_VAE(X_train.shape[1], n_classes, Nb=32, units=500,
                                                                layers_e=2, layers_d=0)
traditional_vae.fit(X_total_input, [X_total, Y_total_input], epochs=50, batch_size=batch_size, verbose=2)

#################### -------Semi-Supervised BAE ---------- ######################

print("\n=====> Evaluate the Models using KNN Search ... \n")

from similarity_search import *

k_topk = 100

p_t, r_t = evaluate_hashing(list_dataset_labels, encoder_Tvae, X_total_input, X_test_input, labels_total, labels_test,
                            traditional=True, tipo="topK")
p_b, r_b = evaluate_hashing(list_dataset_labels, encoder_Bvae, X_total_input, X_test_input, labels_total, labels_test,
                            traditional=False, tipo="topK")

file = open("results/SEMI_Results_Top_K_%s.csv" % dataset_name, "a")
file.write("%s,semi_VDSH, %d, %f, %f, %f\n" % (dataset_name, k_topk, p_t, r_t, test_size))
file.write("%s,%s, %d, %f, %f, %f\n" % (dataset_name, name_model, k_topk, p_b, r_b, test_size))
file.close()

print("DONE ...")

print("\n=====> Evaluate the Models using Range/Ball Search ... \n")

ball_radius = np.arange(0, max_radius)  # ball of radius graphic

binary_p = []
binary_r = []
encode_total = encoder_Bvae.predict(X_total_input)
encode_test = encoder_Bvae.predict(X_test_input)
probas_total = keras.activations.sigmoid(encode_total).eval(session=K.get_session())
probas_test = keras.activations.sigmoid(encode_test).eval(session=K.get_session())
total_hash_b = (probas_total > 0.5) * 1
test_hash_b = (probas_test > 0.5) * 1

traditional_p = []
traditional_r = []
encode_total = encoder_Tvae.predict(X_total_input)
encode_test = encoder_Tvae.predict(X_test_input)
median = MedianHashing()
median.fit(encode_total)
total_hash_t = median.transform(encode_total)
test_hash_t = median.transform(encode_test)

file2 = open("results/SEMI_Results_BallSearch_%s.csv" % dataset_name, "a")

for ball_r in ball_radius:
    test_similares_train = get_similar(test_hash_b, total_hash_b, tipo='ball', ball=ball_r)
    p_b, r_b = measure_metrics(list_dataset_labels, test_similares_train, labels_test, labels_destination=labels_total)

    test_similares_train = get_similar(test_hash_t, total_hash_t, tipo='ball', ball=ball_r)
    p_t, r_t = measure_metrics(list_dataset_labels, test_similares_train, labels_test, labels_destination=labels_total)

    file2.write("%s,semi_VDSH, %d, %f, %f, %f\n" % (dataset_name, ball_r, p_t, r_t, test_size))
    file2.write("%s,%s, %d, %f, %f, %f\n" % (dataset_name, name_model, ball_r, p_b, r_b, test_size))

file2.close()
print("DONE ... ")



# # sBAE3(X_train.shape[1],n_classes,Nb=32,units=500,layers_e=2,layers_d=2)
# # BAE3(data_dim,n_classes,Nb,units,layers_e,layers_d,opt='adam',BN=True):
#
# data_dim = X_train.shape[1]
# n_classes = n_classes
# Nb = 32
# units = 500
# layers_e = 2
# layers_d = 2
# BN = True
# opt = 'adam'
#
#
# def my_model(data_dim, n_classes, Nb, units, layers_e, layers_d, opt='adam', BN=True):
#     pre_encoder = define_pre_encoder(data_dim, layers=layers_e, units=units, BN=BN)
#     generator = define_pre_generator(Nb, data_dim, layers=layers_d, units=units, BN=BN)
#     generator.summary()
#
#     x = Input(shape=(data_dim,))
#
#     hidden = pre_encoder(x)
#     logits_b = Dense(Nb, activation='linear', name='logits-b')(hidden)  # log(B_j/1-B_j)
#
#     def sampling(logits_b):
#         # logits_b = K.log(aux/(1-aux) + K.epsilon() )
#         b = logits_b + sample_gumbel(K.shape(logits_b))  # logits + gumbel noise
#         return keras.activations.sigmoid(b / tau)
#
#     b_sampled = Lambda(sampling, output_shape=(Nb,), name='sampled')(logits_b)
#     hidden_generator = generator(b_sampled)
#
#     output = Dense(data_dim, activation='softmax')(hidden_generator)
#
#     supervised_layer = Dense(n_classes, activation='softmax')(hidden_generator)  # req n_classes
#
#     def sup_gumbel_loss(x, x_hat):
#         reconstruction_loss = keras.losses.categorical_crossentropy(x, x_hat)  # *data_dim
#
#         dist = keras.activations.sigmoid(logits_b)  # B_j = Q(b_j) probability of b_j
#         # by formula
#         kl_disc_loss = Nb * np.log(2) + K.sum(
#             dist * K.log(dist + K.epsilon()) + (1 - dist) * K.log(1 - dist + K.epsilon()),
#             axis=1)
#         # new.. using logits -- second term cannot be simplified
#         # disc_loss = Nb*np.log(2) + K.sum( dist*logits_b + K.log(1-dist + K.epsilon()),axis=1)
#         return K.mean(reconstruction_loss + kl_disc_loss)
#
#     binary_vae = Model(inputs=x, outputs=[output, supervised_layer])
#     binary_vae.compile(optimizer=opt, loss=[sup_gumbel_loss, 'categorical_crossentropy'], loss_weights=[1., 1000.])
#
#     return binary_vae
#
#
# binary_vae = my_model(X_train.shape[1], n_classes, Nb=32, units=500, layers_e=2, layers_d=2)
# # binary_vae.fit([X_total_input,Y_total_input], X_total, epochs=50, batch_size=batch_size,verbose=2)
