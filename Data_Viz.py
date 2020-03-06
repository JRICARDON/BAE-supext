# libraries
from Utils import *

create_dir('results/IMG')
nb = 4

## Unsupervised
data_20news, data_snippets, data_reuters = load_results(type='UNSUP')
plot_results(data_20news, label='20news (Unsupervised)', Nb = nb)
plot_results(data_snippets, label='Snippets (Unsupervised)', Nb = nb)
plot_results(data_reuters, label='Reuters (Unsupervised)', Nb = nb)


## Supervised
data_20news, data_snippets, data_reuters = load_results(type='SUP')
plot_results(data = data_20news, label='20news (Supervised)', Nb = nb)
plot_results(data = data_snippets, label='Snippets (Supervised)', Nb = nb)
plot_results(data = data_reuters, label='Reuters (Supervised)', Nb = nb)


## Semi-Supervised
data_20news, data_snippets, data_reuters = load_results(type='SEMI')
plot_results_semi(data_20news, label = '20news (Semi-Supervised)', type = 'SEMI', Nb = nb)
plot_results_semi(data = data_snippets, label='Snippets (Semi-Supervised)', Nb = nb)
plot_results_semi(data = data_reuters, label='Reuters (Semi-Supervised)', Nb = nb)


## --> Top K

data_20news, data_snippets, data_reuters = load_results_topk()

plot_results_topk(data_20news, label='20news', Nb = nb)
plot_results_topk(data_snippets, label='Snippets', Nb = nb)
plot_results_topk(data_reuters, label='Reuters', Nb = nb)


