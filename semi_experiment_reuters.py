from Utils import *
from semisupervised_models import *
from load_reuters import *
import sys

batch_size = 100
epochs = 50
multilabel = True
max_radius = 30


### ****************** Load Data ****************** ###
### *********************************************** ###

print("\n=====> Loading data ...\n")
dataset_name = "reuters"
texts_t, labels_t, texts_test, labels_test, list_dataset_labels = load_reuters()


labels_t = np.asarray(labels_t)
labels_test = np.asarray(labels_test)



test_size = 10 # int(sys.argv[1])
test_size = test_size/100
texts_train, texts_val, labels_train, labels_val = train_test_split(texts_t, labels_t, random_state=20, test_size=test_size)


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

X_total_input = np.concatenate((X_train_input, X_val_input), axis=0)
X_total = np.concatenate((X_train, X_val), axis=0)


print("\n=====> ENCODING LABELS  ...\n")


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

y_train_input = mlb.fit_transform(labels_train)

y_val_input = mlb.transform(labels_val)

y_zeros = [0 for i in range(y_val_input.shape[1])]
y_val_input_new = np.array([y_zeros for i in range(y_val_input.shape[0])])
y_val_input = y_val_input_new

Y_total_input = np.concatenate((y_train_input,y_val_input),axis=0)
labels_total = np.concatenate((labels_train,labels_val),axis=0)

y_test = mlb.transform(labels_test)

n_classes = Y_total_input.shape[1] #len(list_dataset_labels)



### ****************** Training ****************** ###
### ********************************************** ###

print("\n=====> Creating and Training the Models (VDSH and BAE) ... \n")

traditional_vae, encoder_Tvae, generator_Tvae = traditional_VAE(X_train.shape[1], n_classes, Nb=32, units=500, layers_e=2, layers_d=0)
traditional_vae.fit(X_total_input, [X_total, Y_total_input], epochs=epochs, batch_size=batch_size, verbose=2)


binary_vae, encoder_Bvae, generator_Bvae = sBAE3(X_train.shape[1], n_classes, Nb=32, units=500, layers_e=2,layers_d=2)
binary_vae.fit(X_total_input, [X_total, Y_total_input], epochs=epochs, batch_size=batch_size, verbose=2)
name_model = 'semi_BAE'




print("\n=====> Evaluate the Models using KNN Search ... \n")

from similarity_search import *

k_topk = 100

p_t, r_t = evaluate_hashing(list_dataset_labels, encoder_Tvae, X_total_input, X_test_input, labels_total, labels_test,
                            traditional=True, tipo="topK", multilabel= multilabel)
p_b, r_b = evaluate_hashing(list_dataset_labels, encoder_Bvae, X_total_input, X_test_input, labels_total, labels_test,
                            traditional=False, tipo="topK", multilabel= multilabel)

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
    print(ball_r)
    test_similares_train = get_similar(test_hash_b, total_hash_b, tipo='ball', ball=ball_r)
    p_b, r_b = measure_metrics(list_dataset_labels, test_similares_train, labels_test,
                               labels_destination=labels_total, multilabel= multilabel)

    test_similares_train = get_similar(test_hash_t, total_hash_t, tipo='ball', ball=ball_r)
    p_t, r_t = measure_metrics(list_dataset_labels, test_similares_train, labels_test,
                               labels_destination=labels_total, multilabel= multilabel)

    file2.write("%s,semi_VDSH, %d, %f, %f, %f\n" % (dataset_name, ball_r, p_t, r_t, test_size))
    file2.write("%s,%s, %d, %f, %f, %f\n" % (dataset_name, name_model, ball_r, p_b, r_b, test_size))

file2.close()
print("DONE ... ")

