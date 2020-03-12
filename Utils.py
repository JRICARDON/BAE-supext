import gc,nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from text_representation import *
import numpy as np
import os
# nltk.download('reuters')
# nltk.download('wordnet')

def create_dir (path):
    if not os.path.exists(path):
        print('The directory', path, 'does not exist and will be created')
        os.makedirs(path)
    else:
        print('The directory', path, ' already exists')

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #

def data_in_arrays( loading_function, ratio_val = 0.25):
    # loading_function = load_20news()
    # ratio_val = 0.25
    texts_t, labels_t, texts_test, labels_test, list_dataset_labels = loading_function
    labels_t = np.asarray(labels_t)
    labels_test = np.asarray(labels_test)
    texts_train,texts_val,labels_train,labels_val  = train_test_split(texts_t,labels_t, random_state=20,test_size=ratio_val)
    vectors_train, vectors_val, vectors_test = represent_text(texts_train,texts_val,texts_test,model='TF')
    # print(type(vectors_train))
    X_train = np.asarray(vectors_train.todense())
    X_val = np.asarray(vectors_val.todense())
    X_test = np.asarray(vectors_test.todense())

    del vectors_train,vectors_val,vectors_test
    gc.collect()

    X_train_input = np.log(X_train+1.0)
    X_val_input = np.log(X_val+1.0)
    X_test_input = np.log(X_test+1.0)
    X_raw = [X_train, X_val, X_test]
    X = [X_train_input,X_val_input,X_test_input]
    Y = [labels_train, labels_t, labels_val, labels_test]

    return X_raw, X, Y, list_dataset_labels

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #





def target_in_array(list_dataset_labels, n_classes, labels_train, labels_val, labels_test,
                    multilabel = False, semi_supervised = False):
    from keras.utils import to_categorical
    from sklearn import preprocessing
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer()
    labels_total = np.concatenate((labels_train, labels_val), axis=0)

    if multilabel & semi_supervised:
        y_train_input = mlb.fit_transform(labels_train)
        y_val_input = mlb.transform(labels_val)

        y_zeros = [0 for i in range(y_val_input.shape[1])]
        y_val_input_new = np.array([y_zeros for i in range(y_val_input.shape[0])])
        y_val_input = y_val_input_new

        y_test_input = mlb.transform(labels_test)

    elif multilabel:

        y_tot = mlb.fit_transform(labels_total)
        y_train_input = mlb.transform(labels_train)
        y_val_input = mlb.transform(labels_val)

        y_test_input = mlb.transform(labels_test)

    elif semi_supervised:

        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(list_dataset_labels)


        y_train = label_encoder.transform(labels_train)
        y_val = label_encoder.transform(labels_val)
        y_test = label_encoder.transform(labels_test)

        y_train_input = to_categorical(y_train, num_classes=n_classes)
        y_val_input = to_categorical(y_val, num_classes=n_classes)

        # y_val_input = to_categorical(y_val, num_classes=n_classes)
        y_zeros = [0 for i in range(y_val_input.shape[1])]
        y_val_input_new = np.array([y_zeros for i in range(y_val_input.shape[0])])
        y_val_input = y_val_input_new

        y_test_input = to_categorical(y_test, num_classes=n_classes)

    else:
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(list_dataset_labels)
        y_train = label_encoder.transform(labels_train)
        y_val = label_encoder.transform(labels_val)
        y_test = label_encoder.transform(labels_test)

        y_train_input = to_categorical(y_train,num_classes=n_classes)
        y_val_input = to_categorical(y_val,num_classes=n_classes)
        y_test_input = to_categorical(y_test,num_classes=n_classes)


    Y_total_input = np.concatenate((y_train_input, y_val_input), axis=0)

    return Y_total_input, y_test_input



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #


from similarity_search import *

def save_results(list_dataset_labels, encoder_Tvae, encoder_Bvae,
                 X_total_input, X_test_input, labels_train, labels_total, labels_test,
                 dataset_name, max_radius, K_topK=100, type='UNSUP', multilabel = False, ratio_sup = None, Nb = 32):

    p_t, r_t = evaluate_hashing(list_dataset_labels, encoder_Tvae, X_total_input, X_test_input, labels_total,
                                labels_test, traditional=True, tipo="topK", multilabel = multilabel)
    p_b, r_b = evaluate_hashing(list_dataset_labels, encoder_Bvae, X_total_input, X_test_input, labels_total,
                                labels_test, traditional=False, tipo="topK", multilabel = multilabel)

    file = open("results/" + type + "_Results_Top_K_%s.csv" % dataset_name, "a")

    if type == 'SEMI':
        file.write("%s,VDSH, %d, %f, %f, %f, %d\n" % (dataset_name, K_topK, p_t, r_t, ratio_sup, Nb))
        file.write("%s,BAE, %d, %f, %f, %f, %d\n" % (dataset_name, K_topK, p_b, r_b, ratio_sup, Nb))
        file.close()
        print("DONE ...")
    else:
        file.write("%s,VDSH, %d, %f, %f, %d\n" % (dataset_name, K_topK, p_t, r_t, Nb))
        file.write("%s,BAE, %d, %f, %f, %d\n" % (dataset_name, K_topK, p_b, r_b, Nb))
        file.close()
        print("DONE ...")

    ### ****************** Ball Search Methods ****************** ###

    print("\n=====> Evaluate the Models using Range/Ball Search ... \n")

    encode_total = encoder_Bvae.predict(X_total_input)
    encode_test = encoder_Bvae.predict(X_test_input)
    probas_total = keras.activations.sigmoid(encode_total).eval(session=K.get_session())
    probas_test = keras.activations.sigmoid(encode_test).eval(session=K.get_session())
    total_hash_b = (probas_total > 0.5) * 1
    test_hash_b = (probas_test > 0.5) * 1

    encode_total = encoder_Tvae.predict(X_total_input)
    encode_test = encoder_Tvae.predict(X_test_input)
    median = MedianHashing()
    median.fit(encode_total)
    total_hash_t = median.transform(encode_total)
    test_hash_t = median.transform(encode_test)

    ball_radius = np.arange(0, max_radius)  # ball of radius graphic

    file2 = open("results/" + type + "_Results_BallSearch_%s.csv" % dataset_name, "a")

    for ball_r in ball_radius:
        test_similares_train = get_similar(test_hash_b, total_hash_b, tipo='ball', ball=ball_r)
        p_b, r_b = measure_metrics(list_dataset_labels, test_similares_train, labels_train,
                                   labels_destination=labels_total, multilabel = multilabel)

        test_similares_train = get_similar(test_hash_t, total_hash_t, tipo='ball', ball=ball_r)
        p_t, r_t = measure_metrics(list_dataset_labels, test_similares_train, labels_train,
                                   labels_destination=labels_total, multilabel = multilabel)
        if type == 'SEMI':
            file2.write("%s,VDSH, %d, %f, %f, %f, %d\n" % (dataset_name, ball_r, p_t, r_t, ratio_sup, Nb))
            file2.write("%s,BAE, %d, %f, %f, %f, %d\n" % (dataset_name, ball_r, p_b, r_b, ratio_sup, Nb))
        else:
            file2.write("%s,VDSH, %d, %f, %f, %d\n" % (dataset_name, ball_r, p_t, r_t, Nb))
            file2.write("%s,BAE, %d, %f, %f, %d\n" % (dataset_name, ball_r, p_b, r_b, Nb))

    file2.close()
    print("DONE ... ")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #





### ****************** SUPERVISED PLOTTING ****************** ###

def load_results( type = 'UNSUP' ):
    if type == 'SEMI':
        columns = ['dataset', 'Algorithm', 'balls', 'Precision', 'Recall', 'sup_ratio', 'Nb']
    else:
        columns = ['dataset', 'Algorithm', 'balls', 'Precision', 'Recall', 'Nb']

    # ********** Load data ********** #
    data_20news = pd.read_csv('results/' + type + '_Results_BallSearch_20news.csv', header=None)  # , encoding = 'utf-8')
    data_20news.columns = columns

    data_reuters = pd.read_csv('results/' + type + '_Results_BallSearch_reuters.csv', header=None)  # , encoding = 'utf-8')
    data_reuters.columns = columns

    data_snippets = pd.read_csv('results/'+ type + '_Results_BallSearch_snippets.csv', header=None)  # , encoding = 'utf-8')
    data_snippets.columns = columns

    return data_20news, data_snippets, data_reuters

def plot_results(data, label, type = 'UNSUP', saving = True, Nb = 32):
    # data = data_20news
    import matplotlib.pyplot as plt
    data = data[data.Nb == Nb]
    plt.tight_layout()
    plt.plot(data.balls.unique(), data.Precision[data.Algorithm == 'BAE'], marker='o', color='blue', linewidth=2, label=r"$Prec_{BAE}$")
    plt.plot(data.balls.unique(), data.Precision[data.Algorithm == 'VDSH'], marker='o', color='lightblue', linewidth=2, label=r"$Prec_{VDSH}$")
    plt.plot(data.balls.unique(), data.Recall[data['Algorithm'] == 'BAE'], marker='v', color='red', linewidth=2, label = r"$Recall_{BAE}$")
    plt.plot(data.balls.unique(), data.Recall[data['Algorithm'] == 'VDSH'], marker='v', color='salmon', linewidth=2, label = r"$Recall_{VDSH}$")
    plt.title(label+ ' Nb=' + str(Nb))
    plt.legend()
    plt.xlim((0,10))
    plt.xlabel('balls')
    if saving:
        plt.savefig('results/IMG/' + label + '_' + type + '_Nb=' + str(Nb) + '.png')
    plt.show()
    plt.close()

######################################################################


def plot_results_semi(data, label, type = 'SEMI', threshold = .5, saving = True, Nb = 32):

    import matplotlib.pyplot as plt
    data = data[data.Nb == Nb]

    data = data[data.sup_ratio == threshold]
    plt.tight_layout()
    plt.plot(data.balls.unique(), data.Precision[data.Algorithm == 'BAE'], marker='o', color='blue', linewidth=2, label=r"$Prec_{BAE}$")
    plt.plot(data.balls.unique(), data.Precision[data.Algorithm == 'VDSH'], marker='o', color='lightblue', linewidth=2, label=r"$Prec_{VDSH}$")
    plt.plot(data.balls.unique(), data.Recall[data['Algorithm'] == 'BAE'], marker='v', color='red', linewidth=2, label = r"$Recall_{BAE}$")
    plt.plot(data.balls.unique(), data.Recall[data['Algorithm'] == 'VDSH'], marker='v', color='salmon', linewidth=2, label = r"$Recall_{VDSH}$")
    plt.title(label + ' - sup = ' + str(threshold) + ' Nb=' + str(Nb))
    plt.legend()
    plt.xlabel('balls')
    plt.xlim(0,10)
    if saving:
        plt.savefig('results/IMG/' + label + '_' + type + '_Nb=' + str(Nb) +'.png')
    plt.show()
    plt.close()






def load_results_topk():
    ## SEMI-SUPERVISED
    cols = ['dataset', 'Algorithm', 'K', 'Precision', 'Recall', 'sup_ratio', 'Nb']
    data_20news = pd.read_csv('results/SEMI_Results_Top_K_20news.csv', header=None)  # , encoding = 'utf-8')
    data_20news.columns = cols

    data_reuters = pd.read_csv('results/SEMI_Results_Top_K_reuters.csv', header=None)  # , encoding = 'utf-8')
    data_reuters.columns = cols

    data_snippets = pd.read_csv('results/SEMI_Results_Top_K_snippets.csv', header=None)  # , encoding = 'utf-8')
    data_snippets.columns = cols


    data_20news = data_20news.sort_values('sup_ratio')
    data_snippets = data_snippets.sort_values('sup_ratio')
    data_reuters = data_reuters.sort_values('sup_ratio')

    return data_20news, data_snippets, data_reuters


def plot_results_topk(data, label, saving=True, Nb = 4):
    #data = data_20news
    import matplotlib.pyplot as plt

    data = data[data.Nb == Nb]
    x = data.sup_ratio.unique()
    p_vae = data.Precision[(data.Algorithm == 'VDSH')]
    p_bae = data.Precision[(data['Algorithm'] == 'BAE')]
    r_vae = data.Recall[(data['Algorithm'] == 'VDSH')]
    r_bae = data.Recall[(data['Algorithm'] == 'BAE')]
    plt.tight_layout()
    plt.plot(x, p_bae, marker='o', color='blue', linewidth=2, label=r"$Prec_{BAE}$")
    plt.plot(x, p_vae, marker='o', color='lightblue', linewidth=2, label=r"$Prec_{VDSH}$")
    plt.plot(x, r_bae, marker='v', color='red', linewidth=2, label=r"$Recall_{BAE}$")
    plt.plot(x, r_vae, marker='v', color='salmon', linewidth=2, label=r"$Recall_{VDSH}$")
    plt.title(label + ' (Top K)' + ' Nb=' + str(Nb))
    plt.legend()
    plt.xlabel('Supervision %')
    if saving:
        plt.savefig('results/IMG/' + label + '_Nb=' + str(Nb) + '_top_K_SEMI.png')
    plt.show()
    plt.close()

######################################################################












