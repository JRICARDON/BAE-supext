from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 




#analyzer = TfidfVectorizer(ngram_range=(1, 3)).build_analyzer()
tokenizer = TfidfVectorizer().build_tokenizer()
stemmer = SnowballStemmer("english") 
lemmatizer = WordNetLemmatizer()

"""Extract features from raw input"""
def preProcess(s): #String processor
    return s.lower().strip().strip('-').strip('_')

def number_normalize(doc):
    results = []
    for token in tokenizer(doc):
        token_pro = preProcess(token)
        if len(token_pro) != 0 and not token_pro[0].isdigit():
            results.append(token_pro)
    return results

def stemmed_words(doc):
    results = []
    for token in tokenizer(doc):
        pre_pro = preProcess(token)
        #token_pro = stemmer.stem(pre_pro) #aumenta x10 el tiempo de procesamiento
        token_pro = lemmatizer.lemmatize(pre_pro) #so can explain/interpretae -- aumenta x5 el tiempo de proce
        if len(token_pro) > 2 and not token_pro[0].isdigit(): #elimina palabra largo menor a 2
            results.append(token_pro)
    return results

def get_transform_representation(mode, analizer,min_count,max_feat):
    smooth_idf_b = False
    use_idf_b = False
    binary_b = False

    if mode == 'binary':
        binary_b = True
    elif mode == 'tf':     
        pass #default is tf
    elif mode == 'tf-idf':
        use_idf_b = True
        smooth_idf_b = True #inventa 1 conteo imaginario (como priors)--laplace smoothing
    return TfidfVectorizer(stop_words='english',tokenizer=analizer,min_df=min_count, max_df=0.8, max_features=max_feat
                                ,binary=binary_b, use_idf=use_idf_b, smooth_idf=smooth_idf_b,norm=None
                                  ,ngram_range=(1, 3)) 

def represent_text(texts_train,texts_val,texts_test,model='TF'):

    min_count = 1 #default = 1
    max_feat = 10000 #Best: 10000 -- Hinton (2000)

    if model == 'TF':
        vectorizer = get_transform_representation("tf", stemmed_words,min_count,max_feat)
        print("Using TF Model ... ")
        vectorizer.fit(texts_train)
        vectors_train = vectorizer.transform(texts_train)
        vectors_val = vectorizer.transform(texts_val)
        vectors_test = vectorizer.transform(texts_test)

    else: #'TF-IDF'

        vectorizer2 = get_transform_representation("tf-idf", stemmed_words,min_count,max_feat)
        print("Using TF-IDF Model ... ")
        vectorizer2.fit(texts_train)
        vectors_train = vectorizer2.transform(texts_train)
        vectors_val = vectorizer2.transform(texts_val)
        vectors_test = vectorizer2.transform(texts_test)
        
    print(vectors_train.shape)

    return vectors_train, vectors_val, vectors_test

