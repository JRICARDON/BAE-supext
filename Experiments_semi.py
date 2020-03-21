import gc
from Utils import *
create_dir('results/')

batch_size = 6000
epochs = 2
max_radius = 2
nb = 4


## ------ SEMI-SUPERVISED ------ ##
type_sup = 'SUP_BAE_v1' # sBAE3
supervision_ratios = [.99, 0.9, .7, .5, .3, .1, .01]

type = 'SEMI' # ['UNSUP','SEMI', 'SUP']
semi_supervised = True

for ratio_sup in supervision_ratios:
	exec(open('supervised_VHDS_vs_sBAE3/supervised_exp_20news.py').read())
	gc.collect()


type = 'SEMI' # ['UNSUP','SEMI', 'SUP']
semi_supervised = True

for ratio_sup in supervision_ratios:
	exec(open('supervised_VHDS_vs_sBAE3/supervised_exp_reuters.py').read())
	gc.collect()


type = 'SEMI' # ['UNSUP','SEMI', 'SUP']
semi_supervised = True

for ratio_sup in supervision_ratios:
	exec(open('supervised_VHDS_vs_sBAE3/supervised_exp_snippets.py').read())
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