# libraries
from Utils import *

create_dir('results/IMG')
nb = 64
threshold = .5

## Unsupervised
data_20news, data_snippets, data_reuters = load_results(type='UNSUP')
plot_results(data_20news, label='20news (Unsupervised)', Nb = nb)
plot_results(data_snippets, label='Snippets (Unsupervised)', Nb = nb)
plot_results(data_reuters, label='Reuters (Unsupervised)', Nb = nb)


## Supervised
algorithm = 'sBAE3'
data_20news, data_snippets, data_reuters = load_results(type='SUP')

# data_20news = data_20news.drop_duplicates(subset=['Algorithm', 'Nb', 'balls'])
# data_snippets = data_snippets.drop_duplicates(subset=['Algorithm', 'Nb', 'balls'])
# data_reuters = data_reuters.drop_duplicates(subset=['Algorithm', 'Nb', 'balls'])

plot_results(data = data_20news, label='20news (Sup type 1)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_snippets, label='Snippets (Sup type 1)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_reuters, label='Reuters (Sup type 1)', type='SUP', Nb = nb, type_model=algorithm)


## Supervised
algorithm = 'sBAE4'
data_20news, data_snippets, data_reuters = load_results(type='SUP')

# data_20news = data_20news.drop_duplicates(subset=['Algorithm', 'Nb', 'balls'])
# data_snippets = data_snippets.drop_duplicates(subset=['Algorithm', 'Nb', 'balls'])
# data_reuters = data_reuters.drop_duplicates(subset=['Algorithm', 'Nb', 'balls'])

plot_results(data = data_20news, label='20news (Sup type 2)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_snippets, label='Snippets (Sup type 2)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_reuters, label='Reuters (Sup type 2)', type='SUP', Nb = nb, type_model=algorithm)





## Supervised
algorithm = 'sBAE5'
data_20news, data_snippets, data_reuters = load_results(type='SUP')

# data_20news = data_20news.drop_duplicates(subset=['Algorithm', 'Nb', 'balls'])
# data_snippets = data_snippets.drop_duplicates(subset=['Algorithm', 'Nb', 'balls'])
# data_reuters = data_reuters.drop_duplicates(subset=['Algorithm', 'Nb', 'balls'])

plot_results(data = data_20news, label='20news (Sup type 3)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_snippets, label='Snippets (Sup type 3)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_reuters, label='Reuters (Sup type 3)', type='SUP', Nb = nb, type_model=algorithm)

#####################
## ---- Top K ---- ##


def read_files_topk(type = 'UNSUP'):
    type = 'results/' + type

    data_20news = pd.read_csv(type + '_Results_Top_K_20news.csv',  header=None)
    data_20news.label = '20news'

    data_snippets = pd.read_csv(type + '_Results_Top_K_snippets.csv',  header=None)
    data_snippets.label = 'snippets'

    data_reuters = pd.read_csv(type + '_Results_Top_K_reuters.csv',  header=None)
    data_reuters.label = 'reuters'


    data = pd.concat([data_20news, data_snippets, data_reuters], ignore_index=True, axis= 0, sort=True)
    data.columns = ['Data', 'Algorithm', 'K', 'Precision', 'Recall', 'n_bit']
    return data