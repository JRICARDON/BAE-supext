# libraries
from Utils import *

create_dir('results/IMG')
nb = 32

## Unsupervised
data_20news, data_snippets, data_reuters = load_results(type='UNSUP')
plot_results(data_20news, label='20news (Unsupervised)', Nb = nb)
plot_results(data_snippets, label='Snippets (Unsupervised)', Nb = nb)
plot_results(data_reuters, label='Reuters (Unsupervised)', Nb = nb)


## Supervised
algorithm = 'SUP_BAE_v1'
data_20news, data_snippets, data_reuters = load_results(type='SUP')
plot_results(data = data_20news, label='20news (Sup type 1)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_snippets, label='Snippets (Sup type 1)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_reuters, label='Reuters (Sup type 1)', type='SUP', Nb = nb, type_model=algorithm)


## Supervised
algorithm = 'SUP_BAE_v2'
data_20news, data_snippets, data_reuters = load_results(type='SUP')
plot_results(data = data_20news, label='20news (Sup type 2)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_snippets, label='Snippets (Sup type 2)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_reuters, label='Reuters (Sup type 2)', type='SUP', Nb = nb, type_model=algorithm)



## Supervised
algorithm = 'SUP_BAE_v3'
data_20news, data_snippets, data_reuters = load_results(type='SUP')
plot_results(data = data_20news, label='20news (Sup type 3)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_snippets, label='Snippets (Sup type 3)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_reuters, label='Reuters (Sup type 3)', type='SUP', Nb = nb, type_model=algorithm)


## Semi-Supervised
data_20news, data_snippets, data_reuters = load_results(type='SEMI')
plot_results_semi(data_20news, label = '20news (Semi-Supervised)', type = 'SEMI', Nb = nb, threshold = .3)
plot_results_semi(data = data_snippets, label='Snippets (Semi-Supervised)', type = 'SEMI', Nb = nb, threshold=.3)
plot_results_semi(data = data_reuters, label='Reuters (Semi-Supervised)', type = 'SEMI', Nb = nb, threshold=.3)


## --> Top K

data_20news, data_snippets, data_reuters = load_results_topk()

plot_results_topk(data_20news, label='20news', Nb = nb)
plot_results_topk(data_snippets, label='Snippets', Nb = nb)
plot_results_topk(data_reuters, label='Reuters', Nb = nb)


