import sys
import subprocess
import os

import matplotlib
matplotlib.use('Agg') #must be done first because there is no X server on GP3, etc.
import matplotlib.pyplot as plt

#os.environ['GLOG_minloglevel'] = '2' 
#must be done before importing caffe; prevents tons of screen output during net load

import caffe

if len(sys.argv) != 8:
	print('Usage: python [this.py] [netFull.prototxt] [netReduced.prototxt] [net1.caffemodel] [net2.caffemodel] [num steps] [new prefix] [num test iterations]')
	exit()
netProto = sys.argv[1]
netProtoR = sys.argv[2]
net1Weights = sys.argv[3]
net2Weights = sys.argv[4]
numSteps = int(sys.argv[5])
prefix = sys.argv[6] + '_'
iterations = sys.argv[7]
gpuNum = '0' #the 1080 on GP3
dir_path = os.getcwd() #os.path.dirname(os.path.realpath(__file__))
patternMatch = '] loss = ' #use '] accuracy = ' to record loss instead
minMult = 0.5
maxMult = 1.5


rangeMin = int(numSteps * -1 * minMult)
rangeMax = int(numSteps * maxMult) + 1

caffeHome = '//home//ntopin//caffe-1-3-17//'

print('Loading net 1.')
sys.stdout.flush()
#wait = raw_input("PRESS ENTER TO CONTINUE.")
net1 = caffe.Net(netProtoR, caffe.TEST, weights=net1Weights)
print('Loading net 2.')
sys.stdout.flush()
#wait = raw_input("PRESS ENTER TO CONTINUE.")
net2 = caffe.Net(netProtoR, caffe.TEST, weights=net2Weights)


print('Loading new net.')
sys.stdout.flush()
#wait = raw_input("PRESS ENTER TO CONTINUE.")
newNet = caffe.Net(netProto, caffe.TEST, weights=net1Weights)

for step in range (rangeMin, rangeMax):
	alpha = float(step)/(numSteps)
	print('Processing net for step ' + str(step) + '.')
	sys.stdout.flush()
	#wait = raw_input("PRESS ENTER TO CONTINUE.")
	for param in newNet.params:
		for paramLayer in range(0, len(newNet.params[param])):
			for arrayMember in range(0, len(newNet.params[param][paramLayer].data)):
				newNet.params[param][paramLayer].data[arrayMember] = (1-alpha)*net1.params[param][paramLayer].data[arrayMember] + (alpha)*net2.params[param][paramLayer].data[arrayMember]
	newNet.save(prefix + str(step) + '.caffemodel')

finalOutput = []
for step in range (rangeMin, rangeMax):
	print('Testing for step ' + str(step) + '.')
	sys.stdout.flush()
	#wait = raw_input("PRESS ENTER TO CONTINUE.")
	command = caffeHome + 'build//tools//caffe.bin test -model=' + netProto + ' -weights=' + dir_path + '/' + prefix + str(step) + '.caffemodel -iterations=' + iterations + ' -gpu=' + gpuNum + ' > ' + prefix + str(step) + '_test.txt 2>&1'
	
	p = subprocess.check_output([command], shell=True, stderr=subprocess.STDOUT)

	#text_file = open(prefix + str(step) + '_test.txt', 'w')
	#text_file.write(p)
	#text_file.close()
	
	q = subprocess.Popen(['grep', patternMatch, prefix + str(step) + '_test.txt'], stdout = subprocess.PIPE)
	output2, err2 = q.communicate()
	print(output2)
	#example output2: I0123 14:28:20.243316 10280 caffe.cpp:325] accuracy = 0.9144
	#finding the "=" and cutting it and the space leaves just the number
	finalOutput.append(float(output2[(output2.find('=')+2): (len(output2) if output2.find('(') == -1 else output2.find('('))]))
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_ylabel('Loss')
	ax.set_xlabel('Alpha')
	ax.set_title(prefix[:-1] + ' (with ' + str(int(step + numSteps*minMult)) + ' steps)') # -1 is there to remove added '_'
	ax.plot([float(x) / (numSteps) for x in range(rangeMin, step+1)], finalOutput[:step+1-rangeMin])
	fig.savefig(prefix + 'plot.png')
	plt.close(fig)


text_file = open(prefix + 'final_data.txt', 'w')
text_file.write('Data to plot:\n')
for step in range(rangeMin, rangeMax):
	text_file.write(str(step) + ',' + str(finalOutput[step - rangeMin]) + '\n')
text_file.close()

