# libraries
from Utils import *

create_dir('results/IMG')

## Unsupervised
data_20news, data_snippets, data_reuters = load_results(type='UNSUP')
plot_results(data_20news, label='20news (Unsupervised)')
plot_results(data_snippets, label='Snippets (Unsupervised)')
plot_results(data_reuters, label='Reuters (Unsupervised)')


## Supervised
data_20news, data_snippets, data_reuters = load_results(type='SUP')
plot_results(data = data_20news, label='20news (Supervised)')
plot_results(data = data_snippets, label='Snippets (Supervised)')
plot_results(data = data_reuters, label='Reuters (Supervised)')


## Semi-Supervised
data_20news, data_snippets, data_reuters = load_results(type='SEMI')
plot_results_semi(data_20news, label = '20news (Semi-Supervised)', type = 'SEMI')
plot_results_semi(data = data_20news, label='20news (Semi-Supervised)')
plot_results_semi(data = data_snippets, label='Snippets (Semi-Supervised)')
plot_results_semi(data = data_reuters, label='Reuters (Semi-Supervised)')


## --> Top K

data_20news, data_snippets, data_reuters = load_results_topk()

plot_results_topk(data_20news, label='20news')
plot_results_topk(data_snippets, label='Snippets')
plot_results_topk(data_reuters, label='Reuters')


