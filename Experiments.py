import gc
from Utils import *
create_dir('results/')

batch_size = 100
epochs = 50
nb = 32
max_radius = nb-1

## ------ UNSUPERVISED ------ ##

type = 'UNSUP' #['UNSUP','SEMI', 'SUP']
exec(open('unsupervised_exp_20news.py').read())
gc.collect()

exec(open('unsupervised_exp_reuters.py').read())
gc.collect()

exec(open('unsupervised_exp_snippets.py').read())
gc.collect()



## ----------- SUPERVISED ----------- ##

type_sup = 'SUP_BAE_v1' # sBAE3

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




## ------ SEMI-SUPERVISED ------ ##
type_sup = 'SUP_BAE_v1' # sBAE3
supervision_ratios = [.99, 0.9, .7, .5, .3, .1, .01]

type = 'SEMI' # ['UNSUP','SEMI', 'SUP']
semi_supervised = True

for ratio_sup in supervision_ratios:
	exec(open('supervised_exp_20news.py').read())
	gc.collect()


type = 'SEMI' # ['UNSUP','SEMI', 'SUP']
semi_supervised = True

for ratio_sup in supervision_ratios:
	exec(open('supervised_exp_reuters.py').read())
	gc.collect()


type = 'SEMI' # ['UNSUP','SEMI', 'SUP']
semi_supervised = True

for ratio_sup in supervision_ratios:
	exec(open('supervised_exp_snippets.py').read())
	gc.collect()


