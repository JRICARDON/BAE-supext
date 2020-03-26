# libraries
from Utils import *

create_dir('results/IMG')
nb = 64

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

plot_results(data = data_20news, label='20news (sBAE3)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_snippets, label='Snippets (sBAE3)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_reuters, label='Reuters (sBAE3)', type='SUP', Nb = nb, type_model=algorithm)


## Supervised
algorithm = 'sBAE4'
data_20news, data_snippets, data_reuters = load_results(type='SUP')

# data_20news = data_20news.drop_duplicates(subset=['Algorithm', 'Nb', 'balls'])
# data_snippets = data_snippets.drop_duplicates(subset=['Algorithm', 'Nb', 'balls'])
# data_reuters = data_reuters.drop_duplicates(subset=['Algorithm', 'Nb', 'balls'])

plot_results(data = data_20news, label='20news (sBAE4)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_snippets, label='Snippets (sBAE4)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_reuters, label='Reuters (sBAE4)', type='SUP', Nb = nb, type_model=algorithm)





## Supervised
algorithm = 'sBAE5'
data_20news, data_snippets, data_reuters = load_results(type='SUP')

# data_20news = data_20news.drop_duplicates(subset=['Algorithm', 'Nb', 'balls'])
# data_snippets = data_snippets.drop_duplicates(subset=['Algorithm', 'Nb', 'balls'])
# data_reuters = data_reuters.drop_duplicates(subset=['Algorithm', 'Nb', 'balls'])

plot_results(data = data_20news, label='20news (sBAE5)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_snippets, label='Snippets (sBAE5)', type='SUP', Nb = nb, type_model=algorithm)
plot_results(data = data_reuters, label='Reuters (sBAE5)', type='SUP', Nb = nb, type_model=algorithm)

#####################
## ---- Top K ---- ##


data_20news, data_snippets, data_reuters = load_data_vs_nb()

plot_topk_vs_nb(data_20news, '20news')
plot_topk_vs_nb(data_snippets, 'snippets')
plot_topk_vs_nb(data_reuters, 'reuters')



data_20news, data_snippets, data_reuters = load_data_vs_nb( type = 'SUP')

plot_topk_vs_nb(data_20news, '20news', type = 'SUP')
plot_topk_vs_nb(data_snippets, 'snippets', type = 'SUP')
plot_topk_vs_nb(data_reuters, 'reuters', type = 'SUP')

