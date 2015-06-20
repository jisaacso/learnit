import train
import sys

def main(trainFile, testFile):
    print "Extraining training features"
    d, l = train.loadData(trainFile)
    X, dictionary, tfidf = train.tfidfVectors(d)

    print "Training the model"
    learner = train.trainModel(X, l)

    print "Extracting testing features"
    testd, _ = train.loadData(testFile, _train=False)
    X_test, _, _ = train.tfidfVectors(testd, dictionary, tfidf)

    print "Predicting test classes"
    Y_Pred = learner.predict(X_test)

    print "writing predicted classes"
    with open('predictedLabels', 'w') as fout:
        for pred in Y_Pred:
            pred = str(int(pred))
            fout.write(pred + '\n')

if __name__ == '__main__':
    assert len(sys.argv) == 3,\
        "Usage: python test.py <trainFilePath> <testFilePath>"

    _, trainFile, testFile = sys.argv

    main(trainFile, testFile)
