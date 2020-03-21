import gc
from Utils import *
create_dir('results/')

batch_size = 100
epochs = 50
nb = 32
max_radius = 15

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
semi_supervised = False
ratio_sup = .25
exec(open('supervised_all_exp_20news.py').read())
gc.collect()


type = 'SUP' # ['UNSUP','SEMI', 'SUP']
semi_supervised = False
ratio_sup = .25
exec(open('supervised_all_exp_snippets.py').read())
gc.collect()


type = 'SUP' # ['UNSUP','SEMI', 'SUP']
semi_supervised = False
ratio_sup = .25
exec(open('supervised_all_exp_reuters.py').read())
gc.collect()





