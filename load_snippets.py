

def read_file(archivo,symb=' '):
    with open(archivo,'r') as f:
        lineas = f.readlines()
        tokens_f = [linea.strip().split(symb) for linea in lineas]
        labels = [tokens[-1] for tokens in tokens_f]
        tokens = [' '.join(tokens[:-1]) for tokens in tokens_f]
    return labels,tokens


def load_snippets(root_dir = ''):

	labels_t,texts_t = read_file(root_dir+"Data/data-web-snippets/train.txt")
	labels_test,texts_test = read_file(root_dir+"Data/data-web-snippets/test.txt")
	print("Training data: ",len(texts_t))
	print("Test data: ",len(texts_test))

	labels = list(set(labels_t))

	print("Done!")

	return texts_t, labels_t, texts_test, labels_test, labels

	