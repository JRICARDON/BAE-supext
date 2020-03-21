# libraries
from Utils import *

create_dir('results/IMG')
nb = 64
threshold = .5

## Semi-Supervised
data_20news, data_snippets, data_reuters = load_results(type='SEMI')
plot_results_semi(data_20news, label = '20news (Semi-Supervised)', type = 'SEMI', Nb = nb, threshold = threshold)
plot_results_semi(data = data_snippets, label='Snippets (Semi-Supervised)', type = 'SEMI', Nb = nb, threshold = threshold)
plot_results_semi(data = data_reuters, label='Reuters (Semi-Supervised)', type = 'SEMI', Nb = nb, threshold= threshold)


## --> Top K

data_20news, data_snippets, data_reuters = load_results_topk()

plot_results_topk(data_20news, label='20news', Nb = nb)
plot_results_topk(data_snippets, label='Snippets', Nb = nb)
plot_results_topk(data_reuters, label='Reuters', Nb = nb)

