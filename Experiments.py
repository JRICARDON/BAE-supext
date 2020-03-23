import gc
from Utils import *
create_dir('results/')

batch_size = 6000
epochs = 2
nb = 32
max_radius = 2
ratio_sup = .5
semi_supervised = False

# dataset_name = '20news'


## ------ UNSUPERVISED ------ ##

type = 'UNSUP' #['UNSUP','SEMI', 'SUP']

exec(open('unsupervised_exp_20news.py').read())
gc.collect()

exec(open('unsupervised_exp_reuters.py').read())
gc.collect()

exec(open('unsupervised_exp_snippets.py').read())
gc.collect()


## ----------- SUPERVISED ----------- ##

type = 'SUP' # ['UNSUP','SEMI', 'SUP']

dataset_name = '20news'
exec(open('supervised_experiments.py').read())
gc.collect()

dataset_name = 'snippets'
exec(open('supervised_experiments.py').read())
gc.collect()

dataset_name = 'reuters'
exec(open('supervised_experiments.py').read())
gc.collect()





