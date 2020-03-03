
batch_size = 2000
epochs = 3
max_radius = 3
type = 'UNSUP' #['UNSUP','SEMI', 'SUP']
ratio_sup = .25


exec(open('unsup_experiment_20news.py').read())
exec(open('unsup_experiment_reuters.py').read())
exec(open('unsup_experiment_snippets.py').read())




type = 'SEMI' # ['UNSUP','SEMI', 'SUP']
semi_supervised = True
supervision_ratios = [ .3]

for ratio_sup in supervision_ratios:
	exec(open('supervised_20news.py').read())




type = 'SUP' # ['UNSUP','SEMI', 'SUP']
semi_supervised = False
ratio_sup = .25

exec(open('supervised_20news.py').read())





type = 'SEMI' # ['UNSUP','SEMI', 'SUP']
semi_supervised = True
supervision_ratios = [.7, .5, .3,]

for ratio_sup in supervision_ratios:
	exec(open('supervised_reuters.py').read())


type = 'SUP' # ['UNSUP','SEMI', 'SUP']
semi_supervised = False
ratio_sup = .25
exec(open('supervised_reuters.py').read())



type = 'SEMI' # ['UNSUP','SEMI', 'SUP']
semi_supervised = True
supervision_ratios = [.7, .5, .3,]

for ratio_sup in supervision_ratios:
	exec(open('supervised_snippets.py').read())



type = 'SUP' # ['UNSUP','SEMI', 'SUP']
semi_supervised = False
ratio_sup = .25
exec(open('supervised_snippets.py').read())