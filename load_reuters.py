from nltk.corpus import reuters

def load_reuters():

	documents_stat = reuters.fileids()
	print(str(len(documents_stat)) + " documents")

	train_docs_stat = list(filter(lambda doc: doc.startswith("train"), documents_stat))
	print(str(len(train_docs_stat)) + " total training documents")
	test_docs_stat = list(filter(lambda doc: doc.startswith("test"), documents_stat))
	print(str(len(test_docs_stat)) + " total test documents")

	texts_t = [reuters.raw(archivo) for archivo in train_docs_stat]
	labels_t = [reuters.categories(archivo) for archivo in train_docs_stat]

	texts_test = [reuters.raw(archivo) for archivo in test_docs_stat]
	labels_test = [reuters.categories(archivo) for archivo in test_docs_stat]

	labels = reuters.categories()

	print("Done!")
	return texts_t, labels_t, texts_test, labels_test, labels