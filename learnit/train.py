import numpy as np
from gensim import corpora, models, matutils
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score

MODEL = LogisticRegression(penalty="l2", C=1)

class Labeler():
    def __init__(self):
        self.label2class = {}
        self.maxLabel = 0

    def getLabel(self, label):
        if label in self.label2class:
            return self.label2class[label]
        else:
            self.label2class[label] = self.maxLabel
            self.maxLabel += 1
            return self.label2class[label]
    
    

def loadData(fpath, _train=True):
    """
    Loads Blackbird specified format for training

    Parameters
    ----------
    fpath: string
        Input path for the training or testing features

    _train: boolean
        True if the input file contains training labels

    Returns
    ----------
    docs: list[list[words]]
        A list of documents where each document is represented
        as a list of words

    labels: numpy.array(int)
        An array of class labels
    """
    try:
        with open(fpath, 'r') as fin:
            data = fin.read()
    except IOError:
        print 'Invalid input path %s' % fpath
        return None

    label2class = Labeler()
    lines = data.split('\n')
    labels = []
    docs = []
    for line in lines:
        #words = line.split(' ')
        if _train:
            try:
                label, datum = line.split('\t')
            except ValueError:
                print 'Malformed line %s' % line
                continue
            labels.append(label2class.getLabel(label))
            docs.append(datum.split(' '))
        else:
            docs.append(line.split(' '))
    labels = np.array(labels)
    return docs, labels

    
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


def trainModel(X, Y):
    return MODEL.fit(X, Y)

def testModel(X):
    return MODEL.predict(X)
    
def crossValidateModel(model, X, y):
    """
    Runs a Stratified K Fold cross validation routine. Used
    to prevent overtraining the model

    Parameters
    ----------
    model: sklearn model
        Any scikit learn model implementing .fit and .predict

    X: numpy.array
        A feature matrix

    Y: numpy.array
        A class label vector
    
    Returns
    ----------
    scores: list
        A list of f1 scores for training
    """

    skf = StratifiedKFold(y, 20)
    scores = []
    for train, test in skf:
        model.fit(X[train,:], y[train])
        pred = model.predict(X[test,:])
        print confusion_matrix(y[test], pred)
        scores.append(f1_score(y[test], pred, average='weighted'))
    return scores


def printWeights(model, dictionary):
    """
    Prints the top 10 weights associated with every
    prediction class

    Parameters
    ----------
    model: sklearn model
        Any scikit learn model implementing a .coef (special
        care needs to be taken for SVC models, not supported)

    dictionary: gensim.corpora.dictionary.Dictionary
        A dictionary mapping feature indices to words
    
    """

    nclasses = model.coef_.shape[0]
    ret = []
    for i in range(nclasses):
        labels = np.argsort(np.abs(model.coef_[i,:]))[::-1][:10]
        weights = np.sort(np.abs(model.coef_[i,:]))[::-1][:10]
        ret.append('\n'.join(["%s: %s" %(dictionary[l], w)
                              for l, w in zip(labels, weights)]))
        ret.append('-'*40)
    return '\n'.join(ret)