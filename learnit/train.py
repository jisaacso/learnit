import numpy as np
from gensim import corpora, models, matutils
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from metamind.api import ClassificationModel, set_api_key

def cleanText(text):
    """
    Removes non alphanumeric characters from the input text

    Parameters
    ----------
    text: string
        text to clean

    Returns
    ----------
    string
        Cleaned text
    """

    prevWasBad = False
    ret = []
    for c in text:
        if c.isalnum():
            prevWasBad = False
            ret.append(c)
        else:
            if prevWasBad:
                continue
            else:
                prevWasBad = True
                ret.append(' ')
    return ''.join(ret).lower()

    
class Labeler():
    def __init__(self):
        self.label2class = {}
        self.maxLabel = 0
        self.class2label = {}
        
    def getLabel(self, label):
        if label in self.label2class:
            return self.label2class[label]
        else:
            self.label2class[label] = self.maxLabel
            self.class2label[self.maxLabel] = label
            self.maxLabel += 1
            return self.label2class[label]
    

def loadData(data, _train=True):
    """
    Loads a tsv of text training data

    Parameters
    ----------
    data: string
        One string of data to train on. 

    _train: boolean
        True if the input file contains training labels

    Returns
    ----------
    docs: list[list[words]]
        A list of documents where each document is represented
        as a list of words

    labels: numpy.array(int)
        An array of class labels

    label2class: train.Labeler
        A class containing the mapping from numerical index
        to human input label
    """
    label2class = Labeler()
    lines = data.split('\n')
    labels = []
    docs = []
    for line in lines:
        if _train:
            try:
                label, datum = line.split('\t')
            except ValueError:
                print 'Malformed line %s' % line
                continue
            datum = cleanText(datum)
            labels.append(label2class.getLabel(label))
            docs.append(datum.split(' '))
        else:
            datum = cleanText(line)
            docs.append(datum.split(' '))
    labels = np.array(labels)

    return docs, labels, label2class

    
def tfidfVectors(docs, dictionary=None, tfidf=None):
    """
    Extracts tfidf feature vectors for a given set of
    documents.

    Parameters
    ----------
    docs: list[list[string]]
        List of documents, where each document is represented
        as a list of words

    dictionary: gensim.corpora.dictionary.Dictionary, optional
        This is a dictionary built from the training data. If
        not supplied, the dictionary will be built from all
        documents in docs

    tfidf: gensim.models.tfidfmodel.TfidfModel, optional
        This is a tfidf vector space built from the training data.
        If not supplied, the tfidf vector space will be built from
        the input documents

    Returns
    ----------
    X: numpy.array
        The feature matrix of shape [ndocs, nfeatures]

    dictionary: gensim.corpora.dictionary.Dictionary
        The dictionary defining unique words in X

    tfidf: gensim.models.tfidfmodel.TfidfModel
        The tfidf vector space built by the trainer
    """

    if not dictionary and not tfidf:
        dictionary = corpora.Dictionary(docs)
        corpus = [dictionary.doc2bow(text) for text in docs]
        tfidf = models.TfidfModel(corpus)
    elif not dictionary or not tfidf:
        raise Exception('how?')
    else:
        corpus = [dictionary.doc2bow(text) for text in docs]

    tcorpus = [tfidf[text] for text in corpus]
    X = matutils.corpus2dense(tcorpus, len(dictionary))
    return X.T, dictionary, tfidf


    

def buildKerasNetwork(inputdim, outputdim):
    """
    Using Keras, builds a shallow, fully collected
    neural network with aggressive dropout and a
    softmax output unit.

    Parameters
    ----------
    inputdim: int
        Size of the input unit (no of features)

    outputdim: int
        Size of the output unit (no of classes)

    Returns
    ----------
    model: keras.models.Sequential
        A keras model compiled to contain a fully
        connected neural layer
    """
    model = Sequential()
    model.add(Dense(inputdim, 64, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, outputdim, init='uniform'))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

    
def trainDeepModel(X, Y):
    """
    Rather unfortunately named, this trains the shallow
    neural network defined by buildKerasNetwork

    Parameters
    ----------
    X: np.ndarray
        Features to train the model with. X.shape must
        be [noExamples, featureDim]

    Y: np.array
        Vector of numerical class labels

    Returns
    ----------
    model: keras.models.Sequential
        A keras model trained using the input data
    """
    inputdim = X.shape[1]
    outputdim = np.max(Y) + 1
    numpifiedY = np_utils.to_categorical(Y, nb_classes=outputdim)
    model = buildKerasNetwork(inputdim, outputdim)
    model.fit(X, numpifiedY, validation_split=.2, nb_epoch=20, batch_size=16)
    return model


def testDeepModel(sent, model):
    """
    This evaluates the pre-trained model on the input
    sentence.

    Parameters
    ----------
    sent: str
        Sentence to evaluate the model on

    model: test.MODEL
        Namedtuple containing model parameters (dictionary, tfidf
        learner and labels)

    Returns
    ----------
    predClass: str
        The predicted output class
    """


    X, _, _ = tfidfVectors([sent.split(' ')], model.dictionary, model.tfidf)
    pred = model.learner.predict(X)
    predClass = model.labels.class2label[np.argmax(pred)]
    return predClass

def compareModels(model):
    """
    This evaluates the pre-trained model agaisnt
    metamind's API on sentences in `data/validation`

    Parameters
    ----------
    model: test.MODEL
        Namedtuple containing model parameters (dictionary, tfidf
        learner and labels)

    """
    
    set_api_key("MohJ53r6kUvoPjHS8tStX1vnfssvN5EDetVcp2uCNISwXus2BS")

    with open('data/validation', 'r') as fin:
        validations = fin.read()
        truth = [model.labels.label2class[i] for i in
                 ['positive']*9 + ['negative']*8]          

    scores_mm = []
    scores_joe = []
    for validation in validations.split('\n'):
        mmLabel = testMetaMind(validation)[0]['label']
        scores_mm.append(model.labels.label2class[mmLabel])
        joeLabel = testDeepModel(validation, model)
        scores_joe.append(model.labels.label2class[joeLabel])
        
    print 'MetaMind F1 score is %s' % f1_score(truth, scores_mm)
    print 'My F1 score is %s' % f1_score(truth, scores_joe)
    
    
def testMetaMind(sent):
    return ClassificationModel(id=31638).predict(sent, input_type="text")

def w2vVectors(docs, w2v):
    """ DEPRECATED
    Calculated word2vec features for a set of documents

    Parameters
    ----------
    docs: list[list[str]]
        A list of documents, where each document is a list of words

    w2v: gensim.models.Word2Vec
        A pretrained Word2Vec model

    Returns
    ----------
    ret: np.ndarray
        A feature matrix of size [ndocs, w2vOutputDim]
    """
    networkSize = w2v.syn0.shape[1]
    v_old = None
    ret = np.zeros((len(docs), networkSize))
    for idx, doc in enumerate(docs):
        numWordsInDoc = 0
        for word in doc:
            if not word in w2v.vocab:
                continue
            numWordsInDoc += 1
            v_new = w2v.syn0norm[w2v.vocab[word].index]
            if v_old is None:
                v_old = v_new
                continue
            v_old = np.sum(np.vstack((v_old, v_new)), axis=0)
        if v_old is None:
            print 'v is none'
            v_old = np.zeros(networkSize)
        else:
            v_old /= float(numWordsInDoc)
        ret[idx, :] = v_old
        v_old = None
    return ret
