import gc,nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from text_representation import *
import matplotlib.pyplot as plt
import numpy as np

# nltk.download('reuters')
# nltk.download('wordnet')



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


from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer


def target_in_array(list_dataset_labels, n_classes, labels_train, labels_val, labels_test,
                    multilabel = False, semi_supervised = False):

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

        Y_total_input = mlb.fit_transform(labels_total)
        y_test = mlb.transform(labels_test)

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

    Y_total_input = np.concatenate((y_train_input, y_val_input), axis=0)

    return Y_total_input, y_test_input



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #


from similarity_search import *

def save_results(list_dataset_labels, encoder_Tvae, encoder_Bvae,
                 X_total_input, X_test_input, labels_train, labels_total, labels_test,
                 dataset_name, max_radius, K_topK=100, type='UNSUP', multilabel = False, ratio_sup = None):

    p_t, r_t = evaluate_hashing(list_dataset_labels, encoder_Tvae, X_total_input, X_test_input, labels_total,
                                labels_test, traditional=True, tipo="topK", multilabel = multilabel)
    p_b, r_b = evaluate_hashing(list_dataset_labels, encoder_Bvae, X_total_input, X_test_input, labels_total,
                                labels_test, traditional=False, tipo="topK", multilabel = multilabel)

    file = open("results/" + type + "_Results_Top_K_%s.csv" % dataset_name, "a")

    if type == 'SEMI':
        file.write("%s,VDSH, %d, %f, %f, %f\n" % (dataset_name, K_topK, p_t, r_t, ratio_sup))
        file.write("%s,BAE, %d, %f, %f, %f\n" % (dataset_name, K_topK, p_b, r_b, ratio_sup))
        file.close()
        print("DONE ...")
    else:
        file.write("%s,VDSH, %d, %f, %f\n" % (dataset_name, K_topK, p_t, r_t))
        file.write("%s,BAE, %d, %f, %f\n" % (dataset_name, K_topK, p_b, r_b))
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
            file2.write("%s,VDSH, %d, %f, %f, %f\n" % (dataset_name, ball_r, p_t, r_t, ratio_sup))
            file2.write("%s,BAE, %d, %f, %f, %f\n" % (dataset_name, ball_r, p_b, r_b, ratio_sup))
        else:
            file2.write("%s,VDSH, %d, %f, %f\n" % (dataset_name, ball_r, p_t, r_t))
            file2.write("%s,BAE, %d, %f, %f\n" % (dataset_name, ball_r, p_b, r_b))

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

def load_r_sup():
    #### ----------- SUPERVISED ----------- ####

    # ********** Load data ********** #
    data_SUP_20news = pd.read_csv('results/SUP_Results_BallSearch_20news.csv', header=None)  # , encoding = 'utf-8')
    data_SUP_20news.columns = columns = ['dataset', 'Algorithm', 'balls', 'Precision', 'Recall']

    data_SUP_reuters = pd.read_csv('results/SUP_Results_BallSearch_reuters.csv', header=None)  # , encoding = 'utf-8')
    data_SUP_reuters.columns = columns = ['dataset', 'Algorithm', 'balls', 'Precision', 'Recall']

    data_SUP_snippets = pd.read_csv('results/SUP_Results_BallSearch_snippets.csv', header=None)  # , encoding = 'utf-8')
    data_SUP_snippets.columns = ['dataset', 'Algorithm', 'balls', 'Precision', 'Recall']

    data_SUP_20news.loc[data_SUP_20news.Algorithm == '  BAE', 'Algorithm'] = 'BAE'
    data_SUP_20news.loc[data_SUP_20news.Algorithm == ' sVDSH', 'Algorithm'] = 'sVDSH'

    data_SUP_reuters.loc[data_SUP_reuters.Algorithm == '  BAE', 'Algorithm'] = 'BAE'
    data_SUP_reuters.loc[data_SUP_reuters.Algorithm == ' sVDSH', 'Algorithm'] = 'sVDSH'

    data_SUP_snippets.loc[data_SUP_snippets.Algorithm == '  BAE', 'Algorithm'] = 'BAE'
    data_SUP_snippets.loc[data_SUP_snippets.Algorithm == ' sVDSH', 'Algorithm'] = 'sVDSH'

    return data_SUP_20news, data_SUP_snippets, data_SUP_reuters

def plot_r_sup(data, label, method = 'balls', saving = True):
    plt.tight_layout()
    plt.plot(data.balls.unique(), data.Precision[data.Algorithm == 'BAE'], marker='o', color='blue', linewidth=2, label=r"$Prec_{BAE}$")
    plt.plot(data.balls.unique(), data.Precision[data.Algorithm == 'sVDSH'], marker='o', color='lightblue', linewidth=2, label=r"$Prec_{VDSH}$")
    plt.plot(data.balls.unique(), data.Recall[data['Algorithm'] == 'BAE'], marker='v', color='red', linewidth=2, label = r"$Recall_{BAE}$")
    plt.plot(data.balls.unique(), data.Recall[data['Algorithm'] == 'sVDSH'], marker='v', color='salmon', linewidth=2, label = r"$Recall_{VDSH}$")
    plt.title(label)
    plt.legend()
    plt.xlabel('balls')
    if saving:
        plt.savefig('results/IMG/' + label + '_SUP.png')
    plt.show()
    plt.close()

######################################################################


### ****************** SEMI-SUPERVISED PLOTTING ****************** ###


def load_r_semi():
    ## SEMI-SUPERVISED
    data_SEMI_20news = pd.read_csv('results/SEMI_Results_BallSearch_20news.csv',  header=None) #, encoding = 'utf-8')
    data_SEMI_20news.columns = ['dataset', 'Algorithm', 'balls', 'Precision', 'Recall', 'sup_ratio']

    data_SEMI_reuters = pd.read_csv('results/SEMI_Results_BallSearch_reuters.csv',  header=None, sep=';') #, encoding = 'utf-8')
    data_SEMI_reuters.columns = ['dataset', 'Algorithm', 'balls', 'Precision', 'Recall', 'sup_ratio']

    data_SEMI_snippets = pd.read_csv('results/SEMI_Results_BallSearch_snippets.csv',  header=None) #, encoding = 'utf-8')
    data_SEMI_snippets.columns = ['dataset', 'Algorithm', 'balls', 'Precision', 'Recall', 'sup_ratio']

    data_SEMI_20news.loc[data_SEMI_20news.Algorithm == ' sBAE3_semi', 'Algorithm'] = 'BAE'
    data_SEMI_20news.loc[data_SEMI_20news.Algorithm == ' sVDSH', 'Algorithm'] = 'VDSH'

    data_SEMI_reuters.loc[data_SEMI_reuters.Algorithm == '  BAE', 'Algorithm'] = 'BAE'
    data_SEMI_reuters.loc[data_SEMI_reuters.Algorithm == ' sVDSH', 'Algorithm'] = 'VDSH'

    data_SEMI_snippets.loc[data_SEMI_snippets.Algorithm == ' sBAE3', 'Algorithm'] = 'BAE'
    data_SEMI_snippets.loc[data_SEMI_snippets.Algorithm == ' sVDSH', 'Algorithm'] = 'VDSH'
    return data_SEMI_20news, data_SEMI_snippets, data_SEMI_reuters




def plot_r_semi(data, label, sup_ratio = 0.5, method = 'balls', saving = True):
    x = data.balls.unique()
    p_vae = data.Precision[(data.Algorithm == 'VDSH') & (data.sup_ratio == sup_ratio)]
    p_bae = data.Precision[(data['Algorithm'] == 'BAE') & (data.sup_ratio == sup_ratio)]
    r_vae = data.Recall[(data['Algorithm'] == 'VDSH') & (data.sup_ratio == sup_ratio)]
    r_bae = data.Recall[(data['Algorithm'] == 'BAE') & (data.sup_ratio == sup_ratio)]
    plt.tight_layout()
    plt.plot(x, p_bae, marker='o', color='blue', linewidth=2, label=r"$Prec_{BAE}$")
    plt.plot(x, p_vae, marker='o', color='lightblue', linewidth=2, label=r"$Prec_{VDSH}$")
    plt.plot(x, r_bae, marker='v', color='red', linewidth=2, label=r"$Recall_{BAE}$")
    plt.plot(x, r_vae, marker='v', color='salmon', linewidth=2, label=r"$Recall_{VDSH}$")
    plt.title(label)
    plt.legend()
    plt.xlabel('balls')
    if saving:
        plt.savefig('results/IMG/' + label + '_SEMI.png')
    plt.show()
    plt.close()


def load_topk_semi():
    ## SEMI-SUPERVISED
    cols = ['dataset', 'Algorithm', 'K', 'Precision', 'Recall', 'sup_ratio']
    data_SEMI_20news = pd.read_csv('results/SEMI_Results_Top_K_20news.csv', header=None)  # , encoding = 'utf-8')
    data_SEMI_20news.columns = cols

    data_SEMI_reuters = pd.read_csv('results/SEMI_Results_Top_K_reuters.csv', header=None,
                                    sep=';')  # , encoding = 'utf-8')
    data_SEMI_reuters.columns = cols

    data_SEMI_snippets = pd.read_csv('results/SEMI_Results_Top_K_snippets.csv', header=None)  # , encoding = 'utf-8')
    data_SEMI_snippets.columns = cols

    data_SEMI_20news.loc[data_SEMI_20news.Algorithm == ' sBAE3_semi', 'Algorithm'] = 'BAE'
    data_SEMI_20news.loc[data_SEMI_20news.Algorithm == ' sVDSH', 'Algorithm'] = 'VDSH'

    data_SEMI_reuters.loc[data_SEMI_reuters.Algorithm == '  BAE', 'Algorithm'] = 'BAE'
    data_SEMI_reuters.loc[data_SEMI_reuters.Algorithm == ' sVDSH', 'Algorithm'] = 'VDSH'

    data_SEMI_snippets.loc[data_SEMI_snippets.Algorithm == ' sBAE3', 'Algorithm'] = 'BAE'
    data_SEMI_snippets.loc[data_SEMI_snippets.Algorithm == ' sVDSH', 'Algorithm'] = 'VDSH'

    data_SEMI_20news = data_SEMI_20news.sort_values('sup_ratio')
    data_SEMI_snippets = data_SEMI_snippets.sort_values('sup_ratio')
    data_SEMI_reuters = data_SEMI_reuters.sort_values('sup_ratio')

    return data_SEMI_20news, data_SEMI_snippets, data_SEMI_reuters


def plot_topk_semi(data, label, sup_ratio=0.5, saving=True):
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
    plt.title(label + ' Top K')
    plt.legend()
    plt.xlabel('Supervision %')
    if saving:
        plt.savefig('results/IMG/' + label + '_top_K_SEMI.png')
    plt.show()
    plt.close()

######################################################################


## UNSUPERVISED

def load_r_unsup():
    data_20news = pd.read_csv('results/UNSUP_Results_BallSearch_20news.csv', header=None)  # , encoding = 'utf-8')
    data_20news.columns = columns = ['dataset', 'Algorithm', 'balls', 'Precision', 'Recall']

    data_reuters = pd.read_csv('results/UNSUP_Results_BallSearch_reuters.csv', header=None)
    data_reuters.columns = columns = ['dataset', 'Algorithm', 'balls', 'Precision', 'Recall']

    data_snippets = pd.read_csv('results/UNSUP_Results_BallSearch_snippets.csv', header=None)
    data_snippets.columns = ['dataset', 'Algorithm', 'balls', 'Precision', 'Recall']

    data_snippets.loc[data_snippets.Algorithm == ' BAE', 'Algorithm'] = 'BAE'
    data_snippets.loc[data_snippets.Algorithm == ' VDSH', 'Algorithm'] = 'VDSH'

    data_reuters.loc[data_reuters.Algorithm == ' BAE', 'Algorithm'] = 'BAE'
    data_reuters.loc[data_reuters.Algorithm == ' VDSH', 'Algorithm'] = 'VDSH'

    data_20news.loc[data_20news.Algorithm == ' BAE', 'Algorithm'] = 'BAE'
    data_20news.loc[data_20news.Algorithm == ' VDSH', 'Algorithm'] = 'VDSH'

    return data_20news, data_snippets, data_reuters


def plot_r_unsup(data, label, method='balls', saving=True):
    plt.tight_layout()
    plt.plot(data.balls.unique(), data.Precision[data.Algorithm == 'BAE'], marker='o', color='blue', linewidth=2,
             label=r"$Prec_{BAE}$")
    plt.plot(data.balls.unique(), data.Precision[data.Algorithm == 'VDSH'], marker='o', color='lightblue', linewidth=2,
             label=r"$Prec_{VDSH}$")
    plt.plot(data.balls.unique(), data.Recall[data['Algorithm'] == 'BAE'], marker='v', color='red', linewidth=2,
             label=r"$Recall_{BAE}$")
    plt.plot(data.balls.unique(), data.Recall[data['Algorithm'] == 'VDSH'], marker='v', color='salmon', linewidth=2,
             label=r"$Recall_{VDSH}$")
    plt.title(label)
    plt.legend()
    plt.xlabel('balls')
    if saving:
        plt.savefig('results/IMG/' + label + '_UNSUP.png')
    plt.show()
    plt.close()











