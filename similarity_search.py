import numpy as np
from keras import backend as K
import keras

class MedianHashing(object):
    def __init__(self):
        self.threshold = None
        self.latent_dim = None
    def fit(self, X):
        self.threshold = np.median(X, axis=0)
        self.latent_dim = X.shape[1]
    def transform(self, X):
        assert(X.shape[1] == self.latent_dim)
        binary_code = np.zeros(X.shape)
        for i in range(self.latent_dim):
            binary_code[np.nonzero(X[:,i] < self.threshold[i]),i] = 0
            binary_code[np.nonzero(X[:,i] >= self.threshold[i]),i] = 1
        return binary_code.astype(int)
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
def get_similar(origen,destination,tipo="topK",K=100,ball=2):
    """
        Retrieve similar documents to the origen document inside the destination corpus (source)
    """
    origen_similares = [] #indices
    for number,dato_hash in enumerate(origen):
        hamming_distance = np.sum(dato_hash != destination,axis=1) #distancia de hamming (# bits distintos)
        if tipo=="EM": #match exacto
            ball= 0
        
        if tipo=="ball" or tipo=="EM":
            K = np.sum(np.sort(hamming_distance)<=ball)
            
        #get topK
        ordenados = np.argsort(hamming_distance) #indices
        origen_similares.append(ordenados[:K]) 
        
        origen_similares[-1] = np.setdiff1d(origen_similares[-1] , np.asarray(number))
    return origen_similares

#build similar or compared in labels
def measure_metrics(unique_labels_dataset,data_similars,labels_data,labels_destination=[]):
    #for now compare labels
    count_labels = {label:np.sum([label in aux for aux in labels_data]) for label in unique_labels_dataset}
    precision = 0.
    recall =0.
    for similars, label in zip(data_similars,labels_data): #source de donde se extrajo info
        if len(similars) == 0: #no encontro similares:
            continue
            
        if len(labels_destination) == 0: #extrajo del mismo conjunto
            labels_retrieve = labels_data[similars] 
        else:
            labels_retrieve = labels_destination[similars] 
        
        if type(labels_retrieve[0]) == list: #multiple classes
            tp = np.sum([len(set(label)& set(aux))>=1 for aux in labels_retrieve]) #al menos 1 clase en comun --quizas variar
            recall += tp/np.sum([count_labels[aux] for aux in label ]) #cuenta todos los label del dato
        else: #only one class
            tp = np.sum(labels_retrieve == label) #true positive
            recall += tp/count_labels[label]
        precision += tp/len(similars)
    
    return precision/len(labels_data), recall/len(labels_data)

def evaluate_hashing(unique_labels_dataset,encoder,train,test,labels_trainn,labels_testt,traditional=True,tipo="topK"):
    """
        Evaluate Hashing correclty: Query and retrieve on a different set
    """
    encode_train = encoder.predict(train)
    encode_test = encoder.predict(test)
    if traditional:
        median= MedianHashing()
        median.fit(encode_train)
        
        train_hash = median.transform(encode_train)
        test_hash = median.transform(encode_test)
    else: #para Binary VAE
        probas_train = keras.activations.sigmoid(encode_train).eval(session=K.get_session())
        probas_test = keras.activations.sigmoid(encode_test).eval(session=K.get_session())
        
        train_hash = (probas_train > 0.5)*1
        test_hash = (probas_test > 0.5)*1

    test_similares_train =  get_similar(test_hash,train_hash,tipo="topK",K=100)
    return measure_metrics(unique_labels_dataset,test_similares_train,labels_testt,labels_destination=labels_trainn)


