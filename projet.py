import numpy as np
import random
import math
import time

# robot = 7 arms
armsLengths = [1,1,1,1,1,1,1]

# circle
rb =  4.13 # right big abscissa
lb = -4.13 # left  big abscissa
rs =  3.18 # right small abscissa
ls = -3.18 # left small abscissa
bb = -1.71 # bottom big ordinate
bs = -0.76 # botton small ordinate
b  =  2.42 # bid radian
s  =  1.08 # small radian

#environments
envs = [(lb,b,bb,5+4.47),(ls,s,bs,5+3.36),(rb,b,bb,5+4.47),(rs,s,bs,5+3.36)]

def initNeurons(env):
	neurons = np.ones((7,2), dtype=np.int)
	if env[0] == lb or env[0] == ls:
		neurons[:,0] *= -1
	if env[1] == rb or env[1] == rs:
		neurons[:,1] *= -1
	return neurons.reshape((neurons.size,))

def applyNoise(neurons):
	return neurons * np.concatenate( (np.ones((1,14),dtype=int), np.ones(14,dtype=int) - 2 * np.identity(14,dtype=int)), axis=0 )

def getDesiredAngles(neurons):
	return np.dot( np.array([30,15]), neurons.reshape((len(neurons)/2,2)).T)

def getRandomController(connectionNb):
	connections = np.zeros((14,14), dtype=np.int)
	for i,j in random.sample([(i,j) for i in range(14) for j in range(14) if i!=j ],connectionNb):
		connections[i,j] = np.random.choice([-1,1])
	return connections

def step(neurons,controller):
	return np.where( np.dot(neurons,controller) < 0, -1, 1)

def getXY(angles):
	global armsLengths
	armsLeftFingersAngles = np.cumsum(angles[0:5])
	angles = np.concatenate( (armsLeftFingersAngles , np.cumsum(angles[5:7]) + armsLeftFingersAngles[2]) )
	arms = [ [0.,0.] ]
	for idx,angle in enumerate(angles[0:3]):
		x=arms[idx][0] + armsLengths[idx] * round(math.sin(math.radians(angle)),2)
		y=arms[idx][1] + armsLengths[idx] * round(math.cos(math.radians(angle)),2)
		arms.append([x,y])
	leftFingers = [ list(arms[-1]) ]
	for idx,angle in enumerate(angles[3:5]):
		x=leftFingers[idx][0] + armsLengths[idx+3] * round(math.sin(math.radians(angle)),2)
		y=leftFingers[idx][1] + armsLengths[idx+3] * round(math.cos(math.radians(angle)),2)
		leftFingers.append([x,y])
	rightFingers = [ list(arms[-1]) ]
	for idx,angle in enumerate(angles[5:7]):
		x=rightFingers[idx][0] + armsLengths[idx+5] * round(math.sin(math.radians(angle)),2)
		y=rightFingers[idx][1] + armsLengths[idx+5] * round(math.cos(math.radians(angle)),2)
		rightFingers.append([x,y])
	return arms+leftFingers[1:]+rightFingers[1:]

def fitness(envIdx,neurons):
	global envs
	side,size,hight,Dmax = envs[envIdx]
	xy = getXY(getDesiredAngles(neurons))
	DL = np.abs( np.linalg.norm([xy[5][0]-side,xy[5][1]-hight]) - size)
	DR = np.abs( np.linalg.norm([xy[7][0]-side,xy[7][1]-hight]) - size)
	return 1 - (DL + DR) / Dmax

def score(envsIdx,ic,controllers,startTime):
	scores = np.zeros(len(controllers))
	icTested = 0
	for icIdx,neurons in enumerate(ic):
		for controllerIdx,controller in enumerate(controllers):
			n = np.copy(neurons)
			for w in range(12):
				n = step(neurons,controller)
			scores[controllerIdx] += fitness(envsIdx[icIdx],n)
		icTested += 1
		if np.max(scores) != icTested:
			return scores
	return scores

def evolve(controllers,scores):
	bestControllerIdx = np.argmax(scores)
	for controllerIdx,controller in enumerate(controllers):
		if controllerIdx != bestControllerIdx:
			r = np.sum( controller != 0, axis=0)
			for u,ru in enumerate(r) :
				#node u targeted for change
				if(np.random.rand(1)<=0.05):
					pu = (4*ru)/(3*ru+14)
					#random incoming connection has to be remove
					if np.random.rand(1)<=pu:
						try:
							w = np.random.choice( np.flatnonzero(controller[:,u]) )
							controller[w][u] = 0
						except ValueError:
							pass
					#random incoming connection has to be created
					else:
						try:
							choices = np.ma.masked_array(controller[:,u] == 0)
							choices[u] = np.ma.masked
							choices = np.flatnonzero(choices)
							w = np.random.choice( choices )
							controller[w][u] = np.random.choice([-1,1])
						except ValueError:
							pass
	return controllers

#INITIAL CONDITIONS
# icMat
# [[IC1ENV1, IC2ENV1, ..., IC15ENV1],
#  [IC1ENV2, IC2ENV2, ..., IC15ENV2],
#  [IC1ENV3, IC2ENV3, ..., IC15ENV3],
#  [IC1ENV4, IC2ENV4, ..., IC15ENV4]]
icMat = np.array( [applyNoise(initNeurons(env)) for env in envs] )
# icLin
# [IC1ENV1, IC2ENV1, IC3ENV1, ..., IC15ENV4]
icLin = icMat.reshape((60,14))
# icCol
# [IC1ENV1, IC1ENV2, IC1ENV3, ..., IC15ENV4]
icCol = icMat.T.reshape((14,60)).T

#ENVIRONMENT IDX
envsIdxLin = np.repeat(np.arange(4),14) # [0,0,0,0,1,1,1,1,...,3]
envsIdxCol = np.tile(np.arange(4), 14)  # [0,1,2,3,0,1,2,3,...,3]

#INITIAL RANDOM CONTROLLERS
# 1 controller
# [[ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [ 0, 0, 0, 0, 0,-1, 0, 0, 0, 1, 0, 0, 0, 0],
#  ...
#  [ 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0]]
controllers = np.array([getRandomController(28) for i in range(10) ])

#MAIN LOOP
startTime = time.time()
while True:
	s = score(envsIdxCol,icCol,controllers,startTime)
	if np.max(s) != 60 and (time.time() - startTime) < (2 * 3600):
		controllers = evolve(controllers,s)
	else:
		break