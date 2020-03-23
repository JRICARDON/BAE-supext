import gc
from Utils import *
create_dir('results/')

batch_size = 100
epochs = 50
max_radius = 15
nb = 32


## ------ SEMI-SUPERVISED ------ ##
# type_sup = 'sBAE3'
type = 'SEMI'
supervision_ratios = [.99, .9, .7, .5, .3, .1]
semi_supervised = True

dataset_name = '20news'
for ratio_sup in supervision_ratios:
	exec(open('supervised_experiments.py').read())
	gc.collect()


dataset_name = 'snippets'
for ratio_sup in supervision_ratios:
	exec(open('supervised_experiments.py').read())
	gc.collect()


dataset_name = 'reuters'
for ratio_sup in supervision_ratios:
	print(ratio_sup)
	exec(open('supervised_experiments.py').read())
	gc.collect()




# ## ----------- SUPERVISED ----------- ##
# type_sup = 'SUP_BAE_v1' # sBAE3
#
# type = 'SUP' # ['UNSUP','SEMI', 'SUP']
# semi_supervised = False
# ratio_sup = .25
# exec(open('supervised_VHDS_vs_sBAE3/supervised_exp_20news.py').read())
# gc.collect()
#
#
# type = 'SUP' # ['UNSUP','SEMI', 'SUP']
# semi_supervised = False
# ratio_sup = .25
# exec(open('supervised_VHDS_vs_sBAE3/supervised_exp_reuters.py').read())
# gc.collect()
#
#
# type = 'SUP' # ['UNSUP','SEMI', 'SUP']
# semi_supervised = False
# ratio_sup = .25
# exec(open('supervised_VHDS_vs_sBAE3/supervised_exp_snippets.py').read())
# gc.collect()
#
#
# ## =========> Second type <============= ##
# type_sup = 'SUP_BAE_v2' # sBAE4
#
# type = 'SUP' # ['UNSUP','SEMI', 'SUP']
# semi_supervised = False
# ratio_sup = .25
# exec(open('supervised_VHDS_vs_sBAE3/supervised_exp_20news.py').read())
# gc.collect()
#
#
# type = 'SUP' # ['UNSUP','SEMI', 'SUP']
# semi_supervised = False
# ratio_sup = .25
# exec(open('supervised_VHDS_vs_sBAE3/supervised_exp_reuters.py').read())
# gc.collect()
#
#
# type = 'SUP' # ['UNSUP','SEMI', 'SUP']
# semi_supervised = False
# ratio_sup = .25
# exec(open('supervised_VHDS_vs_sBAE3/supervised_exp_snippets.py').read())
# gc.collect()
#
#
#
# ## =========> Third type <============= ##
# type_sup = 'SUP_BAE_v3' # sBAE3
#
# type = 'SUP' # ['UNSUP','SEMI', 'SUP']
# semi_supervised = False
# ratio_sup = .25
# exec(open('supervised_VHDS_vs_sBAE3/supervised_exp_20news.py').read())
# gc.collect()
#
#
# type = 'SUP' # ['UNSUP','SEMI', 'SUP']
# semi_supervised = False
# ratio_sup = .25
# exec(open('supervised_VHDS_vs_sBAE3/supervised_exp_reuters.py').read())
# gc.collect()
#
#
# type = 'SUP' # ['UNSUP','SEMI', 'SUP']
# semi_supervised = False
# ratio_sup = .25
# exec(open('supervised_VHDS_vs_sBAE3/supervised_exp_snippets.py').read())
# gc.collect()