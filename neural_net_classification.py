from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure import LinearLayer, SigmoidLayer, FullConnection
from pybrain.structure import FeedForwardNetwork

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
from sklearn.metrics             import precision_score,recall_score,confusion_matrix

from collections import defaultdict
from feature_generator import *

num_features = 0
user_comments = defaultdict(list)
users = set([])
for line in open("sampled_users_features_standardized.tsv"):
	user, features = line.strip().split("\t")
	feature_vector = features.strip().split(",")
	feature_vector = [float(x) for x in feature_vector if len(x) > 0][:6]
	if len(users) == 10 and user not in users:
		continue
	users.add(user)
	user_comments[user].append(feature_vector)
	if num_features == 0:
		num_features = len(feature_vector)

user_indices = {}
for i, user in enumerate(list(users)):
	user_indices[user] = i

alldata = ClassificationDataSet(num_features)
for user in user_comments.keys():
	for fv in user_comments[user]:
		alldata.addSample(fv, [user_indices[user]])

tstdata_temp, trndata_temp = alldata.splitWithProportion( 0.25 )

tstdata = ClassificationDataSet(num_features)
for n in xrange(0, tstdata_temp.getLength()):
    tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

trndata = ClassificationDataSet(num_features)
for n in xrange(0, trndata_temp.getLength()):
    trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )
print fnn
# fnn = FeedForwardNetwork()
# inLayer = LinearLayer(trndata.indim)
# hiddenLayer1 = SigmoidLayer(20)
# hiddenLayer2 = SigmoidLayer(10)
# outLayer = LinearLayer(trndata.outdim)
# fnn.addInputModule(inLayer)
# fnn.addModule(hiddenLayer1)
# fnn.addModule(hiddenLayer2)
# fnn.addOutputModule(outLayer)
# in_to_hidden = FullConnection(inLayer, hiddenLayer1)
# hidden_to_hidden = FullConnection(hiddenLayer1, hiddenLayer2)
# hidden_to_out = FullConnection(hiddenLayer2, outLayer)
# fnn.addConnection(in_to_hidden)
# fnn.addConnection(hidden_to_hidden)
# fnn.addConnection(hidden_to_out)
# fnn.sortModules()

trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

def checkNeuralNet(dataset):
    predictedVals=trainer.testOnClassData(dataset=dataset)
    actualVals=list(dataset['class'])
    print("-----------------------------")
    print("-----------------------------")
    print "Micro precision:", precision_score(actualVals,predictedVals,average='micro')
    print "Micro recall:", recall_score(actualVals,predictedVals,average='micro')
    print "Macro precision:", precision_score(actualVals,predictedVals,average='macro')
    print "Macro recall:", recall_score(actualVals,predictedVals,average='macro')
    print("The confusion matrix is as shown below:")
    print(confusion_matrix(actualVals,predictedVals))

for i in range(80):
	trainer.trainEpochs(5)
	# trnresult = percentError(trainer.testOnClassData(), trndata['class'])
	# tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
	# print "epoch: %4d" % trainer.totalepochs
	# print "train error: %5.2f%%" % trnresult
	# print "test error: %5.2f%%" % tstresult
	print "train data"
	checkNeuralNet(trndata)
	print "test data"
	checkNeuralNet(tstdata)

