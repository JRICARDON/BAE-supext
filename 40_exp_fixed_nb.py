import gc
from Utils import *
create_dir('results/')

batch_size = 100
epochs = 50
max_radius = 15
nb = 4

ratio_sup = .5
semi_supervised = False



## ------ UNSUPERVISED ------ ##
type = 'UNSUP' #['UNSUP','SEMI', 'SUP']

dataset_name = '20news'
exec(open('unsupervised_experiments.py').read())
gc.collect()

dataset_name = 'snippets'
exec(open('unsupervised_experiments.py').read())
gc.collect()

dataset_name = 'reuters'
exec(open('unsupervised_experiments.py').read())
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


