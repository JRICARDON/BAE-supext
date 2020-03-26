import gc
from Utils import *
create_dir('results/')

batch_size = 100
epochs = 50
max_radius = 15
nb = 4



## ------ SEMI-SUPERVISED ------ ##
type = 'SEMI'
supervision_ratios = [.99, .9, .7, .5, .3, .1, .01]
semi_supervised = True

nb_all = [4, 8, 16, 32, 64]

for nb in nb_all:

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




