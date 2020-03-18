import gc
from Utils import *
create_dir('results/')

batch_size = 3000
epochs = 3
max_radius = 5
nb = 4


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


## =========> Second type <============= ##
type_sup = 'SUP_BAE_v2' # sBAE4

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



## =========> Third type <============= ##
type_sup = 'SUP_BAE_v3' # sBAE3

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