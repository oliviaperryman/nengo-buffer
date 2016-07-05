import matplotlib.pyplot as plt
import nengo
import numpy as np
import scipy.ndimage
import matplotlib.animation as animation
from matplotlib import pylab
from PIL import Image
import nengo.spa as spa
import cPickle
import random

from nengo_extras.data import load_mnist
from nengo_extras.vision import Gabor, Mask

dim =28

# --- load the data
img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = load_mnist()

X_train = 2 * X_train - 1  # normalize to -1 to 1
X_test = 2 * X_test - 1  # normalize to -1 to 1

def intense(img):
    newImg = img.copy()
    newImg[newImg < 0] = -1
    newImg[newImg > 0] = 1
    return newImg

#Create set of noisy images	
noise_train = np.random.random(X_train.shape)
noise_train = 2 * noise_train -1 # normalize to -1 to 1

#Clean up noisy images with intensifying
clean_train = intense(noise_train)

rng = np.random.RandomState(9)

# --- set up network parameters
n_vis = noise_train.shape[1]
n_out =  noise_train.shape[1]
#number of neurons/dimensions of semantic pointer
n_hid = 5000 #Try with more neurons for more accuracy


#Want the encoding/decoding done on the training images
ens_params = dict(
    eval_points=X_train,
    neuron_type=nengo.LIF(), #Why not use LIF? originally used LIFRate()
    intercepts=nengo.dists.Choice([-0.5]),
    max_rates=nengo.dists.Choice([100]),
    )


#Least-squares solver with L2 regularization.
solver = nengo.solvers.LstsqL2(reg=0.01)
#solver = nengo.solvers.LstsqL2(reg=0.0001)


#network that generates the weight matrices between neuron activity and images and the labels
with nengo.Network(seed=3) as model:
    a = nengo.Ensemble(n_hid, n_vis, seed=3, **ens_params)
    v = nengo.Node(size_in=n_out)
    conn = nengo.Connection(
        a, v, synapse=None,
        eval_points=X_train, function=X_train,
        solver=solver)
    

# linear filter used for edge detection as encoders, more plausible for human visual system
encoders = Gabor().generate(n_hid, (11, 11), rng=rng)
encoders = Mask((28, 28)).populate(encoders, rng=rng, flatten=True)
#Set the ensembles encoders to this
a.encoders = encoders

#Get the neuron activity of an image or group of images (this is the semantic pointer in this case)
def get_activities(sim, images):
    _, acts = nengo.utils.ensemble.tuning_curves(a, sim, inputs=images)
    return acts
	
#Get the representation of the image after it has gone through the encoders (Gabor filters) but before it is in the neurons
#This must be computed to create the weight matrix for rotation from neuron activity to this step
# This allows a recurrent connection to be made from the neurons to themselves later
def get_encoder_outputs(sim,images):
    #Pass the images through the encoders
    outs = np.dot(images,sim.data[a].encoders.T) #before the neurons 
    return outs


with nengo.Simulator(model) as sim:    
    
    #Neuron activities 
    noise_acts = get_activities(sim,noise_train)
    clean = get_encoder_outputs(sim,clean_train)

    #solvers for a learning rule
    solver_low_pass = nengo.solvers.LstsqL2(reg=1e-8)

    #find weight matrix between neuron activity of the original image and the clean img
    #weights returns a tuple including information about learning process, just want the weight matrix
    weights,_ = solver_low_pass(noise_acts, clean)
	
filename = "low_pass_weights" + str(n_hid) +".p"
cPickle.dump(weights, open( filename, "wb" ) )