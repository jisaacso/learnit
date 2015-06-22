import tornado.ioloop
import tornado.web
import argparse
from io import BytesIO
import sys
from test import buildModel, evalModel

class Mutator():
    def __init__(self):
        self.messageToDisplay = 'Please upload some data'
        self.model = None

class MainHandler(tornado.web.RequestHandler):
    """
    The main page calls get() on this handler. It pulls the next
    test page, labels it and passes it down to the iframe handler
    """
    def initialize(self, mutator):
        self.mutator = mutator
        
    def get(self):
        self.render('templates/index.html',
                    message=self.mutator.messageToDisplay)

        

class DatazHandler(tornado.web.RequestHandler):
    def initialize(self, mutator):
        self.mutator = mutator

    def post(self):
        print "Training the model"
        sentence = self.request.arguments['testInput'][0]
        #predClass = evalModel(sentence, self.mutator.model)

        try:
            predClass = evalModel(sentence, self.mutator.model)
        except AttributeError:
            self.mutator.messageToDisplay = "Please upload data and submit"+\
                                            " before running a test example"
            self.redirect('/', status=303)

        self.mutator.messageToDisplay = "Model predicts %s" % predClass
        self.redirect('/', status=303)

        
class UploadHandler(tornado.web.RequestHandler):
    def initialize(self, mutator):
        self.mutator = mutator


    def post(self):
        print "uploading file"
        trainingFile = self.request.files['fileName'][0]['body']
        print "data received"
        print "building model"
        model = buildModel(trainingFile)
        self.mutator.model = model
        self.mutator.messageToDisplay = "Model has been successfully trained"
        self.redirect('/', status=303)

      
def main(port):
    mutator = Mutator()
    
    application = tornado.web.Application([
        (r"/", MainHandler, {'mutator': mutator}),
        (r"/static/(.+)", tornado.web.StaticFileHandler, {'path': './static'}),
        (r"/dataz", DatazHandler, {'mutator': mutator}),
        (r"/uploader", UploadHandler, {'mutator': mutator}),
    ])

    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    port = int(sys.argv[1] )
    main(port)
