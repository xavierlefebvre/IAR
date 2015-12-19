import numpy as np
import random
import math


# robot
# 7 bras
# controller = 14 neurones + connections

# position du cercle
l=-10  # left        (abscisse)
r=10   # right       (abscisse)
lh=-10 # left hight  (ordonnee)
rh=-10 # right hight (ordonnee)
# rayon du cercle
b=4    # big
s=2    # small

def initNeurons(env):
	neurons = np.ones((7,2), dtype=np.int)
	if env[0] == l:
		neurons[:,0] *= -1
	if env[1] == b:
		neurons[:,1] *= -1
	return neurons.reshape((neurons.size,))

def applyNoise(neurons):
	return neurons * np.concatenate( (np.ones((1,14),dtype=int), np.ones(14,dtype=int) - 2 * np.identity(14,dtype=int)), axis=0 )

def getDesiredAngles(neurons):
	return np.dot( np.array([30,15]), neurons.reshape((len(neurons)/2,2)).T)

def getRandomController(neurons,connectionNb):
	connections = np.zeros((len(neurons),len(neurons)), dtype=np.int)
	for i,j in random.sample([(i,j) for i in range(len(neurons)) for j in range(len(neurons)) if i!=j ],connectionNb):
		connections[i,j] = np.random.choice([-1,1])
	return (neurons,connections)

def run(controller):
	neurons,connections = controller
	return(np.where( np.dot(neurons,connections) < 0, -1, 1),connections)

def getXY(armsLengths,angles):
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
		output.append( getDesiredAngles(np.array(neurons)) )
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
		[0,-45,-45,-45,-45,45,-45]]
	expected = [
		[ [0.,0.], [0.,1.], [0.,2.], [0.,3.], [0.,4.], [0.,5.], [0.,4.], [0.,5.] ],
		[ [0.,0.], [0.,1.], [-0.71,1.71], [-1.71,1.71], [-2.42,1.], [-2.42,0.], [-2.42,2.42], [-3.42,2.42]] ]
	output = []
	for angles in anglesInput:
		output.append( getXY(armsLenghtsInput,angles) )
	if np.array_equal(expected,output):
		print("OK \t getXY")
	else:
		print("ERROR \t getXY")

runTests()

envs = [(l,b),(l,s),(r,b),(r,s)]
ic = np.array( [applyNoise(initNeurons(env)) for env in envs] )