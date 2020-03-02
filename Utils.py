import gc,nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from text_representation import *
import matplotlib.pyplot as plt
import numpy as np

nltk.download('reuters')
nltk.download('wordnet')



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











