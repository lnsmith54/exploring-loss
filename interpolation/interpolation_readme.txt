How to use interpolation code:
"interpolationEvaluation.py" is a simple script for creating and evaluating networks created through linear weight interpolation between two *.caffemodel files. It is used as follows:
python [this.py] [netFull.prototxt] [netReduced.prototxt] [net1.caffemodel] [net2.caffemodel] [num steps] [new prefix] [num test iterations]

where:
"this.py" = path to location of script
"netFull.prototxt" = path to original prototxt (with input layers)
"netReduced.prototxt" = path to prototxt without input layers (deploy prototxt)
"net1.caffemodel" = path to first *.caffemodel file used for interpolation
"net2.caffemodel" = path to second *.caffemodel file used for interpolation
"num steps" = number of points between alpha=0 and alpha=1
"new prefix" = prefix for output *.caffemodel files
"num test iterations" = number of passes to use to evaluate an interpolation

Within interpolationEvaluation.py are variables whose value could/should be changed. "caffeHome" must be set to the path of your caffe root directory.

Optional changes include:
"gpuNum" specifies which gpus to use by ID number. Set to '0' by default in order to use a specific gpu on a multi-gpu machine we used.
"minMult"/"maxMult" specify the magnitude for the lowest and highest alpha values, respectively. Set to 0.5 and 1.5 in order to evalute alpha values in the range -0.5 to 1.5.
"patternMatch" determines which statistic is graphed (i.e., loss or accuracy).

The script first generates the networks, using the specified prefix to generate names, then evaluates each in turn. The results of testing are stored and the relevant number is extracted from the log file. These are then plotted after each evaluation so one can easily observe the progress made. The extracted data is saved separately should one want to plot it with a different method.

"interpolateMany.sh" is a script for performing interpolations between various snapshots of the same initial run. Note the many hard-coded paths. These should be changed to one's own *.prototxt files and *.caffemodel naming patterns. We used the pattern "[some text]_[iteration number].caffemodel" where "iteration number" was a multiple of 5000 between 5000 and 100000. This script, as set up, performed interpolations between the weights during iteration X and iteration Y, where X + delta*5000 = Y, Y <= 100000, and X >= 5000, for a variety of delta values.