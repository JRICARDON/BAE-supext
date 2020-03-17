from Utils import *
from unsupervised_models import *
from load_20news import *

# batch_size = 3000
# epochs = 3
# max_radius = 5
# nb = 32
# type = 'UNSUP' #['UNSUP','SEMI', 'SUP']
# ratio_sup = .25

dataset_name = '20news'
multilabel = False



### ****************** Loading and Transforming ****************** ###

X_raw, X, Y, list_dataset_labels = data_in_arrays(load_20news())

X_train_input, X_val_input, X_test_input = X[0], X[1], X[2]
labels_train, labels_t, labels_val, labels_test = Y[0], Y[1], Y[2], Y[3]
X_train, X_val, X_test = X_raw[0], X_raw[1], X_raw[2]

X_total_input = np.concatenate((X_train_input,X_val_input),axis=0)
X_total = np.concatenate((X_train,X_val),axis=0)
labels_total = np.concatenate((labels_train,labels_val),axis=0)


### *********************** Modeling *********************** ###

print("\n=====> Creating and Training the Models (VDSH and BAE) ... \n")
traditional_vae,encoder_Tvae,generator_Tvae = traditional_VAE(X_train.shape[1],Nb=nb,units=500,layers_e=2,layers_d=0)
traditional_vae.fit(X_total_input, X_total, epochs=epochs, batch_size=batch_size,verbose=2)

binary_vae,encoder_Bvae,generator_Bvae = binary_VAE(X_train.shape[1],Nb=nb,units=500,layers_e=2,layers_d=2)
binary_vae.fit(X_total_input, X_total, epochs=epochs, batch_size=batch_size,verbose=2)




### ****************** Saving ******************* ###

save_results(list_dataset_labels, encoder_Tvae, encoder_Bvae,
			 X_total_input, X_test_input, labels_train, labels_total, labels_test,
			 dataset_name, max_radius,
			 K_topK = 100, type = type, multilabel = multilabel, Nb=nb )



from similarity_search import *

# # def save_results(list_dataset_labels, encoder_Tvae, encoder_Bvae,
# #                  X_total_input, X_test_input, labels_train, labels_total, labels_test,
# #                  dataset_name, max_radius, K_topK=100, type='UNSUP', multilabel = False, ratio_sup = None, Nb = 32):
#
# list_dataset_labels
# encoder_Tvae
# encoder_Bvae
# X_total_input
# X_test_input
# labels_train # target variable of the training set n=8485
# labels_total # target variable of all train data (train + val)  n=11314
# labels_test # target variable of the test set 7532
# dataset_name
# max_radius
# K_topK = 100
# type = 'UNSUP'
# multilabel = False
# ratio_sup = None
# Nb = 32
#
#     p_t, r_t = evaluate_hashing(list_dataset_labels, encoder_Tvae, X_total_input, X_test_input, labels_total,
#                                 labels_test, traditional=True, tipo="topK", multilabel = multilabel)
#     p_b, r_b = evaluate_hashing(list_dataset_labels, encoder_Bvae, X_total_input, X_test_input, labels_total,
#                                 labels_test, traditional=False, tipo="topK", multilabel = multilabel)
#
#     file = open("results/" + type + "_Results_Top_K_%s.csv" % dataset_name, "a")
#
#     if type == 'SEMI':
#         file.write("%s,VDSH, %d, %f, %f, %f, %d\n" % (dataset_name, K_topK, p_t, r_t, ratio_sup, Nb))
#         file.write("%s,BAE, %d, %f, %f, %f, %d\n" % (dataset_name, K_topK, p_b, r_b, ratio_sup, Nb))
#         file.close()
#         print("DONE ...")
#     else:
#         file.write("%s,VDSH, %d, %f, %f, %d\n" % (dataset_name, K_topK, p_t, r_t, Nb))
#         file.write("%s,BAE, %d, %f, %f, %d\n" % (dataset_name, K_topK, p_b, r_b, Nb))
#         file.close()
#         print("DONE ...")
#
#     ### ****************** Ball Search Methods ****************** ###
#
#     print("\n=====> Evaluate the Models using Range/Ball Search ... \n")
#
#     encode_total = encoder_Bvae.predict(X_total_input)
#     encode_test = encoder_Bvae.predict(X_test_input)
#     probas_total = keras.activations.sigmoid(encode_total).eval(session=K.get_session())
#     probas_test = keras.activations.sigmoid(encode_test).eval(session=K.get_session())
#     total_hash_b = (probas_total > 0.5) * 1
#     test_hash_b = (probas_test > 0.5) * 1
#
#     encode_total = encoder_Tvae.predict(X_total_input)
#     encode_test = encoder_Tvae.predict(X_test_input)
#     median = MedianHashing()
#     median.fit(encode_total)
#     total_hash_t = median.transform(encode_total)
#     test_hash_t = median.transform(encode_test)
#
#     ball_radius = np.arange(0, max_radius)  # ball of radius graphic
#
#     file2 = open("results/" + type + "_Results_BallSearch_%s.csv" % dataset_name, "a")
#
#     for ball_r in ball_radius:
#         ball_r = 2
#         test_similares_train = get_similar(test_hash_b, total_hash_b, tipo='ball', ball=ball_r)
#         p_b, r_b = measure_metrics(list_dataset_labels, test_similares_train, labels_train,
#                                    labels_destination=labels_total, multilabel = multilabel)
#
#         # measure_metrics(unique_labels_dataset, data_similars, labels_data, labels_destination=[], multilabel=False):
#         unique_labels_dataset = list_dataset_labels
#         data_similars = test_similares_train
#         labels_data = labels_test
#         labels_destination = labels_total
#         # Counting of the  number of observation in each class in the training set
#         if multilabel:
#             count_labels = {label: np.sum([label in aux for aux in labels_destination]) for label in
#                             unique_labels_dataset}
#         else:
#             count_labels = {label: np.sum([label == aux for aux in labels_destination]) for label in
#                             unique_labels_dataset}
#
#         precision = 0.
#         recall = 0.
#
#         # Consider the label and the similars of all the observations in the test set
#         for similars, label in zip(data_similars, labels_data):
#             if len(similars) == 0:  # There is no similar objects
#                 continue
#
#             # if len(labels_destination) == 0:
#             #     labels_retrieve = labels_data[similars]
#             # else:
#
#             # Retrieving the labels of the obs. in the training set that are similar to the obs. of the test set
#             labels_retrieve = labels_destination[similars]
#
#             if multilabel == True:  # multiple classes
#                 # considering all the train obs. that have at least one label in common with the test obs.
#                 tp = np.sum([len(set(label) & set(aux)) >= 1 for aux in labels_retrieve])
#                 # computing the recall by considering all the labels in the training set
#                 recall += tp / np.sum([count_labels[aux] for aux in label])
#             else:  # only one class
#                 tp = np.sum(labels_retrieve == label)  # true positive
#                 recall += tp / count_labels[label]
#             precision += tp / len(similars)
#
#         total_precision = precision / len(labels_data)
#         total_recall = recall / len(labels_data)
#
#         return total_precision, total_recall
#
#
#
#
#         test_similares_train = get_similar(test_hash_t, total_hash_t, tipo='ball', ball=ball_r)
#         p_t, r_t = measure_metrics(list_dataset_labels, test_similares_train, labels_train,
#                                    labels_destination=labels_total, multilabel = multilabel)
#         if type == 'SEMI':
#             file2.write("%s,VDSH, %d, %f, %f, %f, %d\n" % (dataset_name, ball_r, p_t, r_t, ratio_sup, Nb))
#             file2.write("%s,BAE, %d, %f, %f, %f, %d\n" % (dataset_name, ball_r, p_b, r_b, ratio_sup, Nb))
#         else:
#             file2.write("%s,VDSH, %d, %f, %f, %d\n" % (dataset_name, ball_r, p_t, r_t, Nb))
#             file2.write("%s,BAE, %d, %f, %f, %d\n" % (dataset_name, ball_r, p_b, r_b, Nb))
#
#     file2.close()
#     print("DONE ... ")