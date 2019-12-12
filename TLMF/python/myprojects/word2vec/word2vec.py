from preprocessing import get_data
import sys
# Import and download stopwords from NLTK.
from nltk.corpus import stopwords
from nltk import download

# download('stopwords')  # Download stopwords list.
# Import word2vec model
from gensim.models import Word2Vec
import gensim as gensim

# Remove stopwords.
stop_words = stopwords.words('english')

# Download data for tokenizer.
from nltk import word_tokenize

from sklearn.decomposition import PCA
from matplotlib import pyplot

# download('punkt')
DATASET_DIR = "../../scattered_content/"
ORIGIN_MODEL = "~/Documents/Word2Vec/GoogleNews-vectors-negative300.bin"


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


# Pre-processing a document.
def preprocess_gensim(doc):
    """ preprocess raw text by tokenising and removing stop-words,special-charaters """
    doc = doc.lower()  # Lower the text.
    doc = word_tokenize(doc)  # Split into words.
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
    return doc


# Train a word2vec model with default vector size of 100
def train_word2vec(train_data, worker_no=-1, vector_size=300, model_name="word2vec_model"):
    """ Trains a word2vec model on the preprocessed data and saves it . """
    if not train_data:
        print "no training data"
        return

    w2v_corpus = [preprocess_gensim(train_data[i]) for i in range(len(train_data))]

    print (w2v_corpus[0])
    exit()
    # print (len(w2v_corpus))

    print ("init model")

    model = Word2Vec(w2v_corpus, workers=worker_no, size=vector_size, sg=1, window=6, iter=10, ns_exponent=0.75,
                     compute_loss=True, min_alpha=0.005, min_count=1)

    # print ("build vocab")
    # model.build_vocab(w2v_corpus)

    print ("load google news model")

    # model.intersect_word2vec_format(ORIGIN_MODEL, lockf=1.0, binary=True)
    # model.wv.add()

    INTER_EPOCH = 100

    for epoch in range(INTER_EPOCH):
        print ("epoch :" + str(epoch))
        print (model.get_latest_training_loss())
        print ("loss :" + str(model.get_latest_training_loss()))

        model.compute_loss = True
        model.train(w2v_corpus, total_words=model.corpus_count, epochs=model.epochs)
        model.alpha = model.alpha * 0.9
        model.min_alpha = model.alpha


    # model.save(model_name)
    # model = gensim.models.KeyedVectors.load_word2vec_format(ORIGIN_MODEL, binary=True)
    # model.add()
    # model.train(w2v_corpus, workers=4)

    model.save(model_name)
    print "Model Created Successfully"


# Load the Model
def load_model(path="word2vec_model"):
    """ loads the stored  word2vec model """
    name = Word2Vec.load(path)
    return name


if __name__ == "__main__":
    train_data = get_data(DATASET_DIR)
    train_word2vec(train_data)
