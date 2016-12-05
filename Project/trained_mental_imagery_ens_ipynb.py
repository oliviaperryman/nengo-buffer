import nengo
import numpy as np
import cPickle
from nengo_extras.data import load_mnist
from nengo_extras.vision import Gabor, Mask
from matplotlib import pylab
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import random
import scipy.ndimage

# --- load the data
img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = load_mnist()

X_train = 2 * X_train - 1  # normalize to -1 to 1
X_test = 2 * X_test - 1  # normalize to -1 to 1

temp = np.diag([1]*10)

ZERO = temp[0]
ONE =  temp[1]
TWO =  temp[2]
THREE= temp[3]
FOUR = temp[4]
FIVE = temp[5]
SIX =  temp[6]
SEVEN =temp[7]
EIGHT= temp[8]
NINE = temp[9]

labels =[ZERO,ONE,TWO,THREE,FOUR,FIVE,SIX,SEVEN,EIGHT,NINE]

dim =28

label_weights = cPickle.load(open("label_weights5000.p", "rb"))
activity_to_img_weights = cPickle.load(open("activity_to_img_weights5000.p", "rb"))
identity_after_encoder_weights =  cPickle.load(open("identity_after_encoder_weights5000.p", "r"))
#rotated_after_encoder_weights_5000 =  cPickle.load(open("rotated_after_encoder_weights_5000.p", "r"))

#rotation_weights = cPickle.load(open("rotation_weights_clockwise5000.p","rb"))

#label_weights = cPickle.load(open("label_weights_rot_enc5000.p", "rb"))
#activity_to_img_weights = cPickle.load(open("activity_to_img_weights_rot_enc5000.p", "r"))
#rotated_after_encoder_weights =  cPickle.load(open("rotated_counter_after_encoder_weights_rot_enc5000.p", "r"))

rng = np.random.RandomState(9)
n_hid = 5000
model = nengo.Network(seed=3)
with model:
    #Stimulus only shows for brief period of time
    stim = nengo.Node(lambda t: THREE if t < 0.1 else 0) #nengo.processes.PresentInput(labels,1))#
    
    ens_params = dict(
        eval_points=X_train,
        neuron_type=nengo.LIF(),
        intercepts=nengo.dists.Choice([-0.5]),
        max_rates=nengo.dists.Choice([100]),
        )
        
    
    # linear filter used for edge detection as encoders, more plausible for human visual system
    encoders = Gabor().generate(n_hid, (11, 11), rng=rng)
    encoders = Mask((28, 28)).populate(encoders, rng=rng, flatten=True)

    
    ens = nengo.Ensemble(n_hid, dim**2, seed=3, encoders=encoders, **ens_params)
    
    #Recurrent connection on the neurons of the ensemble to maintain the image
    nengo.Connection(ens.neurons, ens.neurons, transform = identity_after_encoder_weights.T, synapse=0.2)      
    #Can't just connect neurons to neurons
    #nengo.Connection(ens.neurons, ens.neurons, synapse=0.2)      

    #Connect stimulus to ensemble, transform using learned weight matrices
    nengo.Connection(stim, ens, transform = np.dot(label_weights,activity_to_img_weights).T, synapse=0.1)
    
    #Collect output, use synapse for smoothing
    probe = nengo.Probe(ens.neurons,synapse=0.1)
   
sim = nengo.Simulator(model)
sim.run(5)

filename = "mental_imagery_output_"  + str(n_hid) + ".p"
cPickle.dump(sim.data[probe], open( filename , "wb" ) )

