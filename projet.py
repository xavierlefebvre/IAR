import numpy as np
import random
import math

#position du cercle (left/right)
#rayon/diametre du cercle (big/small)
#forme du robot : 7 longueurs
#position du robot : 14 angles
#controller = 14 neurones + connections
#h=-10 # (hight) : ordonnee du centre du cercle

l=-10 # left
r=10  # right
b=4   # big
s=2   # small

def initNeurons(env):
	neurons = np.ones((7,2), dtype=np.int)
	if env[0] == l:
		neurons[:,0] = np.multiply(-1,neurons[:,0])
	if env[1] == b:
		neurons[:,1] = np.multiply(-1,neurons[:,1])
	return neurons.reshape((neurons.size,)).tolist()

def getDesiredAngles(neurons):
	angles = []
	for dof in np.array(neurons).reshape((len(neurons)/2,2)):
		if   np.array_equal(dof,[-1,-1]):
			angles.append(-45)
		elif np.array_equal(dof,[-1, 1]):
			angles.append(-15)
		elif np.array_equal(dof,[ 1,-1]):
			angles.append( 15)
		elif np.array_equal(dof,[ 1, 1]):
			angles.append( 45)
	return angles

def getXY(armsLengths,angles):
	angles = np.cumsum(angles)
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

def getInitialPercepts(envs):
	initConditions = np.array([[ initNeurons(env) for env in envs ]])
	for i in range(14):
		noisyPercepts = np.copy(initConditions[0])
		noisyPercepts[:,i] = - noisyPercepts[:,i]
		initConditions = np.concatenate( (initConditions,[noisyPercepts]) )
	return initConditions

def createRandomNetwork(neurons,connectionNb):
	connections = []
	while connectionNb > 0:
		neuronsList = np.arange(len(neurons)).tolist()
		origin = random.choice(neuronsList)
		neuronsList.remove(origin)
		destination = random.choice(neuronsList)
		if not( (origin,destination) in connections):
			connections.append([origin,destination])
			connectionNb -= 1
	return (neurons,connections)

def runTests():
	#initNeuronsTest
	envsInput = [(l,b),(l,s),(r,b),(r,s)]
	expected = [
		[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
		[-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1],
		[ 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],
		[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
	output = []
	for env in envsInput:
		output.append( initNeurons(env) )
	if np.array_equal(expected,output):
		print("OK \t initNeurons")
	else:
		print("ERROR \t initNeurons")

	#getDesiredAnglesTest
	neuronsInput = [
		[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
		[-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1],
		[ 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],
		[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
	expected = [
		[-45,-45,-45,-45,-45,-45,-45],
		[-15,-15,-15,-15,-15,-15,-15],
		[ 15, 15, 15, 15, 15, 15, 15],
		[ 45, 45, 45, 45, 45, 45, 45]]
	output = []
	for neurons in neuronsInput:
		output.append( getDesiredAngles(neurons) )
	if np.array_equal(expected,output):
		print("OK \t getDesiredAngles")
	else:
		print("ERROR \t getDesiredAngles")
		print(expected)
		print(output)

	#getXYTest
	armsLenghtsInput = [1,1,1,1,1,1,1]
	anglesInput = [
		[0,0,0,0,0,0,0],
		[0,-45,-45,-45,-45,45,45]]
	expected = [
		[ [0.,0.], [0.,1.], [0.,2.], [0.,3.], [0.,4.], [0.,5.], [0.,4.], [0.,5.] ],
		[[0.0, 0.0], [0.0, 1.0], [-0.71, 1.71], [-1.71, 1.71], [-2.42, 1.0], [-2.42, 0.0], [-2.42, 1.0], [-3.42, 1.0]]]
	output = []
	for angles in anglesInput:
		output.append( getXY(armsLenghtsInput,angles) )
	if np.array_equal(expected,output):
		print("OK \t getXY")
	else:
		print("ERROR \t getXY")

runTests()
