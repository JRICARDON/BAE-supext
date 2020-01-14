import numpy as np
import keras
from keras.layers import *
from keras.models import Sequential,Model
from keras import backend as K

def define_pre_encoder(data_dim,layers=2,units=512,dropout=0.0,BN=False): #define pre_encoder network
    model = Sequential(name='pre-encoder')
    model.add(InputLayer(input_shape=(data_dim,)))
    for i in range(1,layers+1):
        #model.add(Dense(int(units/i), activation='relu'))
        model.add(Dense(units,activation='relu'))
        if dropout != 0. and dropout != None:
            model.add(Dropout(dropout))
        if BN:
            model.add(BatchNormalization())
    return model

def define_generator(Nb,data_dim,layers=2,units=32,dropout=0.0,BN=False,exclusive=True):
    model = Sequential(name='generator/decoder')
    model.add(InputLayer(input_shape=(Nb,)))
    for i in np.arange(layers,0,-1):
        #model.add(Dense(int(units/i), activation='relu'))
        model.add(Dense(units,activation='relu'))
        if dropout != 0. and dropout != None:
            model.add(Dropout(dropout))
        if BN:
            model.add(BatchNormalization())
    if exclusive:
        model.add(Dense(data_dim, activation='softmax')) #softmax generator
    else:
        model.add(Dense(data_dim, activation='sigmoid'))
    return model

def traditional_VAE(data_dim,Nb,units,layers_e,layers_d,opt='adam',BN=True):
    pre_encoder = define_pre_encoder(data_dim, layers=layers_e,units=units,BN=BN)
    print("pre-encoder network:")
    pre_encoder.summary()
    generator = define_generator(Nb,data_dim,layers=layers_d,units=units,BN=BN)
    print("generator network:")
    generator.summary()
    
    ## Encoder
    x = Input(shape=(data_dim,))
    hidden = pre_encoder(x)
    z_mean = Dense(Nb,activation='linear', name='z-mean')(hidden)
    z_log_var = Dense(Nb,activation='linear',name = 'z-log_var')(hidden)
    encoder = Model(x, z_mean) # build a model to project inputs on the latent space

    def sampling(args):
        epsilon_std = 1.0
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], Nb),mean=0., stddev=epsilon_std)
        return z_mean + K.exp(0.5*z_log_var) * epsilon #+sigma (desvest)
    
    ## Decoder
    z_sampled = Lambda(sampling, output_shape=(Nb,), name='sampled')([z_mean, z_log_var])
    output = generator(z_sampled)

    def vae_loss(x, x_hat):
        reconstruction_loss = keras.losses.categorical_crossentropy(x, x_hat)*data_dim 
        #reconstruction_loss = keras.losses.binary_crossentropy(x, x_hat)*data_dim 

        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) #con varianza
        return K.mean(reconstruction_loss  + kl_loss)

    traditional_vae = Model(x, output)
    traditional_vae.compile(optimizer=opt,loss=vae_loss)
    return traditional_vae,encoder,generator


tau = K.variable(0.67, name="temperature") #o tau fijo en 0.67=2/3

anneal_rate = 0.003
min_temperature = 0.5

def sample_gumbel(shape,eps=K.epsilon()):
    """Inverse Sample function from Gumbel(0, 1)"""
    U = K.random_uniform(shape, 0, 1)
    return K.log(U + eps)- K.log(1-U + eps)

class My_Callback(keras.callbacks.Callback):    
    def on_epoch_end(self, epoch, logs={}):
        K.set_value(tau_ann, np.max([K.get_value(tau_ann) * np.exp(- anneal_rate * epoch), min_temperature])) 
        print(tau_ann.value().eval(session=keras.backend.get_session()))
        return

def binary_VAE(data_dim,Nb,units,layers_e,layers_d,opt='adam',BN=True):
    pre_encoder = define_pre_encoder(data_dim, layers=layers_e,units=units,BN=BN)
    print("pre-encoder network:")
    pre_encoder.summary()
    generator = define_generator(Nb,data_dim,layers=layers_d,units=units,BN=BN)
    print("generator network:")
    generator.summary()

    x = Input(shape=(data_dim,))
    hidden = pre_encoder(x)
    logits_b  = Dense(Nb, activation='linear', name='logits-b')(hidden) #log(B_j/1-B_j)
    #proba = np.exp(logits_b)/(1+np.exp(logits_b)) = sigmoidal(logits_b) <<<<<<<<<< recupera probabilidad
    #dist = Dense(Nb, activation='sigmoid')(hidden) #p(b) #otra forma de modelarlo
    encoder = Model(x, logits_b)

    def sampling(logits_b):
        #logits_b = K.log(aux/(1-aux) + K.epsilon() )
        b = logits_b + sample_gumbel(K.shape(logits_b)) # logits + gumbel noise
        return keras.activations.sigmoid( b/tau )

    b_sampled = Lambda(sampling, output_shape=(Nb,), name='sampled')(logits_b)
    output = generator(b_sampled)

    def gumbel_loss(x, x_hat):
        reconstruction_loss = keras.losses.categorical_crossentropy(x, x_hat)*data_dim

        dist = keras.activations.sigmoid(logits_b) #B_j = Q(b_j) probability of b_j
        #by formula
        kl_disc_loss = Nb*np.log(2) + K.sum( dist*K.log(dist + K.epsilon()) + (1-dist)* K.log(1-dist + K.epsilon()),axis=1)
        # new.. using logits -- second term cannot be simplified
        #kl_disc_loss = Nb*np.log(2) + K.sum( dist*logits_b + K.log(1-dist + K.epsilon()),axis=1)
        return K.mean(reconstruction_loss  + kl_disc_loss)
    
    binary_vae = Model(x, output)
    binary_vae.compile(optimizer=opt, loss=gumbel_loss)
    return binary_vae, encoder,generator



