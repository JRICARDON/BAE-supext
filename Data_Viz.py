# libraries
from Utils import *

## Supervised
sup_20news, sup_snippets, sup_reuters = load_r_sup()
plot_r_sup(data=sup_20news, label='20news (Supervised)')
plot_r_sup(data=sup_snippets, label='snippets (Supervised)')
plot_r_sup(data=sup_reuters, label='reuters (Supervised)')



## Semi-supervised
semi_20news, semi_snippets, semi_reuters = load_r_semi()

plot_r_semi(data = semi_20news, label = '20_news (Supervision 50%)', sup_ratio=0.5)
plot_r_semi(data = semi_snippets, label = 'snippets (Supervision 50%)', sup_ratio=0.5)
plot_r_semi(data = semi_reuters, label = 'reuters (Supervision 50%)', sup_ratio=0.5)

## --> Top K

semi_20news, semi_snippets, semi_reuters = load_topk_semi()

plot_topk_semi(semi_20news, label='20news')
plot_topk_semi(semi_snippets, label='snippets')
plot_topk_semi(semi_reuters, label='reuters')



## Unsupervised
unsup_20news, unsup_snippets, unsup_reuters = load_r_unsup()

plot_r_unsup(data=unsup_20news, label='20news (Unsupervised)')
plot_r_unsup(data=unsup_snippets, label='snippets (Unsupervised)')
plot_r_unsup(data=unsup_reuters, label='reuter (Unsupervised)')
