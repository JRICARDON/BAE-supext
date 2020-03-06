import gc
from Utils import *
create_dir('results/')

batch_size = 3000
epochs = 3
max_radius = 5
nb = 4
supervision_ratios = [.7, .3,]


## ------ UNSUPERVISED ------ ##

type = 'UNSUP' #['UNSUP','SEMI', 'SUP']
ratio_sup = .25
exec(open('unsupervised_exp_20news.py').read())
gc.collect()

exec(open('unsupervised_exp_reuters.py').read())
gc.collect()

exec(open('unsupervised_exp_snippets.py').read())
gc.collect()




## ------ SUPERVISED ------ ##

type = 'SUP' # ['UNSUP','SEMI', 'SUP']
semi_supervised = False
ratio_sup = .25
exec(open('supervised_exp_20news.py').read())
gc.collect()

type = 'SUP' # ['UNSUP','SEMI', 'SUP']
semi_supervised = False
ratio_sup = .25
exec(open('supervised_exp_reuters.py').read())
gc.collect()


type = 'SUP' # ['UNSUP','SEMI', 'SUP']
semi_supervised = False
ratio_sup = .25
exec(open('supervised_exp_snippets.py').read())
gc.collect()



