from Utils import *
from supervised_models import *
from load_snippets import *

dataset_name = '20news'
batch_size = 5000
epochs = 2
multilabel = False
nb = 4
max_radius = 1
#
# #['UNSUP','SEMI', 'SUP']
type = 'SUP'
ratio_sup = .5


dataset_name = 'snippets'
multilabel = False
semi_supervised = False

### ****************** Loading and Transforming ****************** ###

X_raw, X, Y, list_dataset_labels = data_in_arrays(load_snippets(), ratio_val=ratio_sup)

X_train_input, X_val_input, X_test_input = X[0], X[1], X[2]
labels_train, labels_t, labels_val, labels_test = Y[0], Y[1], Y[2], Y[3]
X_train, X_val, X_test = X_raw[0], X_raw[1], X_raw[2]

X_total_input = np.concatenate((X_train_input,X_val_input),axis=0)
X_total = np.concatenate((X_train,X_val),axis=0)


print("\n=====> Encoding Labels ...\n")

n_classes = len(list_dataset_labels)
labels_total = np.concatenate((labels_train, labels_val), axis=0)

Y_total_input, y_test_input = target_in_array(list_dataset_labels, n_classes,
											  labels_train, labels_val, labels_test,
											  multilabel = False, semi_supervised = semi_supervised)


print("\n=====> Creating and Training the Models (VDSH and BAE) ... \n")

traditional_vae,encoder_Tvae,generator_Tvae = traditional_VAE(X_train.shape[1],n_classes,Nb=nb,units=500,layers_e=2,layers_d=0)
traditional_vae.fit(X_total_input, [X_total, Y_total_input], epochs=epochs, batch_size=batch_size,verbose=2)

# if type_sup == 'SUP_BAE_v1':
binary_vae_v1, encoder_Bvae_v1, generator_Bvae_v1 = sBAE3(X_train.shape[1], n_classes, Nb=nb, units=500, layers_e=2, layers_d=2)
binary_vae_v1.fit(X_total_input, [X_total, Y_total_input], epochs=epochs, batch_size=batch_size, verbose=2)
# elif type_sup == 'SUP_BAE_v2':
binary_vae_v2, encoder_Bvae_v2, generator_Bvae_v2 = sBAE4(X_train.shape[1], n_classes, Nb=nb, units=500, layers_e=2, layers_d=2)
binary_vae_v2.fit([X_total_input, Y_total_input], X_total, epochs=epochs, batch_size=batch_size, verbose=2)
# elif type_sup == 'SUP_BAE_v3':
binary_vae_v3, encoder_Bvae_v3, generator_Bvae_v3 = sBAE5(X_train.shape[1], n_classes, Nb=nb, units=500, layers_e=2, layers_d=2)
binary_vae_v3.fit([X_total_input, Y_total_input], X_total, epochs=epochs, batch_size=batch_size, verbose=2)


# model = 'VDSH'
save_single_model(list_dataset_labels, X_total_input, X_test_input, labels_train, labels_total, labels_test,
                  encoder = encoder_Tvae,dataset_name = dataset_name, K_topK = 100, max_radius = max_radius,
                  type = type, multilabel = multilabel, ratio_sup = ratio_sup, Nb=nb)

model = 'sBAE3'
save_single_model(list_dataset_labels, X_total_input, X_test_input, labels_train, labels_total, labels_test,
                  encoder = encoder_Bvae_v1, dataset_name = dataset_name, model_label = model, K_topK = 100,
                  max_radius = max_radius, type = type, multilabel = multilabel, ratio_sup = ratio_sup, Nb=nb)

model = 'sBAE4'
save_single_model(list_dataset_labels, X_total_input, X_test_input, labels_train, labels_total, labels_test,
                  encoder = encoder_Bvae_v2, dataset_name = dataset_name, model_label = model, K_topK = 100,
                  max_radius = max_radius, type = type, multilabel = multilabel, ratio_sup = ratio_sup, Nb=nb)


model = 'sBAE5'
save_single_model(list_dataset_labels, X_total_input, X_test_input, labels_train, labels_total, labels_test,
                  encoder = encoder_Bvae_v3, dataset_name = dataset_name, model_label = model, K_topK = 100,
                  max_radius = max_radius, type = type, multilabel = multilabel, ratio_sup = ratio_sup, Nb=nb)

