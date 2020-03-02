#### 20 NEWSGROUPS
from sklearn.datasets import fetch_20newsgroups

def clean_20news(textos):

	## specific clean on this dataset:
	headers_to_save = ["Subject","Summary","Organization","Keywords"]

	aux = []

	for textito in textos:

		header,texto = textito.split("\n\n",1)

        #dic = {line.split(": ")[0]:line.split(": ")[1] for line in header.split("\n") }

		dic = {}

		for line in header.split("\n"):
			try:
				a,b = line.split(": ",1)
				dic[a] = b
			except:
				continue
		to_add = ""
		for t in headers_to_save:
			try:
				to_add += dic[t] +". "
			except:
				continue
		aux.append(to_add+texto)

	return aux

def load_20news():

	newsgroups_t = fetch_20newsgroups(subset='train')
	newsgroups_test = fetch_20newsgroups(subset='test')
	labels = newsgroups_t.target_names

	texts_t = newsgroups_t.data
	y_t = newsgroups_t.target
	labels_t = [labels[valor] for valor in y_t]

	texts_test = newsgroups_test.data
	y_test = newsgroups_test.target
	labels_test = [labels[valor] for valor in y_test]

	print("Datos de entrenamiento: ",y_t.shape)
	print("Datos de prueba: ",y_test.shape)

	texts_t = clean_20news(texts_t)
	texts_test = clean_20news(texts_test)

	print("Done!")
	return texts_t, labels_t, texts_test, labels_test, labels

