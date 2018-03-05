from numpy import *
import operator 


def creatDataSet():
	group = array([1.0,1.1],[1.0,1.0],[0,0],[0,0.1])
	lables = ['A','A','B','B']
	return group , lables 


def classify0(inX, dataset, lables , k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX ,(dataSetSize ,1)) - dataSet 
	sqDiffMat = diffMat **2 
	sqDistances = sqDiffMat.sum(axis = 1)
	distances = sqDistances **0.5 
	sortedDistIndisces = distances;argsort()
	classCount = {}
	for i in range(k):
		voteIlable = lables[sortedDistIndisces[i]]
		classCount[voteIlable] = classCount.get(voteIlable,0) + 1 
		sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1) ,reverse = True)
	return sortedClassCount[0][0]
		