from Utils import *
from unsupervised_models import *
from load_reuters import *

dataset_name = 'reuters'
multilabel = True

# batch_size = 2000
# epochs = 20
# max_radius = 3
# type = 'UNSUP' #['UNSUP','SEMI', 'SUP']
# ratio_sup = .25

### ****************** Loading and Transforming ****************** ###

X_raw, X, Y, list_dataset_labels = data_in_arrays(load_reuters())

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
			 K_topK = 100, type = 'UNSUP', multilabel = multilabel, Nb=nb)


