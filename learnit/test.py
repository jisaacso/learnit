import train
import sys
from collections import namedtuple
import numpy as np
MODEL = namedtuple('MODEL', 'learner labels dictionary tfidf')

def buildModel(trainData):
    """
    Builds a machine learner to best fit the input text data

    Parameters
    ----------
    trainData: string
        Input data for training the learner. Should be tab separated
        in the form <LABEL>\t<WORDS>

    Returns
    ----------
    m: MODEL
        A named tuple consisting of objects needed for feature
        transformation (dictionary, tfidf) and learning (learner,
        labels)
    """

    print "Extraining training features"
    d, l, labels = train.loadData(trainData)
    X, dictionary, tfidf = train.tfidfVectors(d)

    print "Training the model"
    learner = train.trainDeepModel(X, l)
    #learner = train.buildKerasNetwork(X.shape[1], labels.maxLabel)
    #learner.load_weights('data/learner.pkl')
    m = MODEL(learner=learner,
              labels = labels,
              dictionary=dictionary,
              tfidf=tfidf)
    return m
    
def evalModel(sentence, model):
    """
    Given a pretrained model, this will apply the model to the input
    sentence.

    Parameters
    ----------
    sentence: string
        sentence of words to evaluate the model on

    model: MODEL
        namedtuple of model parameters as described above

    Returns
    ----------
    classPred: string
        The name of the class which was predicted
    """
    
    classPred = train.testDeepModel(sentence, model)
    return classPred