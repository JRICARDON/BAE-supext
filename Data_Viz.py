# libraries
from Utils import *

## Supervised
sup_20news, sup_snippets, sup_reuters = load_r_sup()
plot_r_sup(data=sup_20news, label='IMG/20news')
plot_r_sup(data=sup_snippets, label='IMG/snippets')
plot_r_sup(data=sup_reuters, label='IMG/reuter')



## Semi-supervised
semi_20news, semi_snippets, semi_reuters = load_r_semi()

plot_r_semi(data = semi_20news, label = 'IMG/20_news', sup_ratio=0.5)
plot_r_semi(data = semi_snippets, label = 'IMG/snippets', sup_ratio=0.5)
plot_r_semi(data = semi_reuters, label = 'IMG/reuters', sup_ratio=0.5)

## --> Top K

semi_20news, semi_snippets, semi_reuters = load_topk_semi()

plot_topk_semi(semi_20news, label='IMG/20news')
plot_topk_semi(semi_snippets, label='IMG/snippets')
plot_topk_semi(semi_reuters, label='IMG/reuters')



## Unsupervised
unsup_20news, unsup_snippets, unsup_reuters = load_r_unsup()

plot_r_unsup(data=unsup_20news, label='IMG/20news')
plot_r_unsup(data=unsup_snippets, label='IMG/snippets')
plot_r_unsup(data=unsup_reuters, label='IMG/reuter')
