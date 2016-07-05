#import matplotlib.pyplot as plt
#%matplotlib inline
import nengo
import numpy as np
import scipy.ndimage
#import matplotlib.animation as animation
#from matplotlib import pylab
from PIL import Image
import nengo.spa as spa
import cPickle
import random

from nengo_extras.data import load_mnist
from nengo_extras.vision import Gabor, Mask

#Encode categorical integer features using a one-hot aka one-of-K scheme.
def one_hot(labels, c=None):
    assert labels.ndim == 1
    n = labels.shape[0]
    c = len(np.unique(labels)) if c is None else c
    y = np.zeros((n, c))
    y[np.arange(n), labels] = 1
    return y
	
# --- load the data
img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = load_mnist()

X_train = 2 * X_train - 1  # normalize to -1 to 1
X_test = 2 * X_test - 1  # normalize to -1 to 1

train_targets = one_hot(y_train, 10)
test_targets = one_hot(y_test, 10)

rng = np.random.RandomState(9)

# --- set up network parameters
#Want to encode and decode the image
n_vis = X_train.shape[1]
n_out =  X_train.shape[1]
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
solver2 = nengo.solvers.LstsqL2(reg=0.01)

#network that generates the weight matrices between neuron activity and images and the labels
with nengo.Network(seed=3) as model:
    a = nengo.Ensemble(n_hid, n_vis, seed=3, **ens_params)
    v = nengo.Node(size_in=n_out)
    conn = nengo.Connection(
        a, v, synapse=None,
        eval_points=X_train, function=X_train,#want the same thing out (identity)
        solver=solver)
    
    v2 = nengo.Node(size_in=train_targets.shape[1])
    conn2 = nengo.Connection(
        a, v2, synapse=None,
        eval_points=X_train, function=train_targets, #Want to get the labels out
        solver=solver2)
    

# linear filter used for edge detection as encoders, more plausible for human visual system
encoders = Gabor().generate(n_hid, (11, 11), rng=rng)
encoders = Mask((28, 28)).populate(encoders, rng=rng, flatten=True)
#Set the ensembles encoders to this
a.encoders = encoders

#Get the one hot labels for the images
def get_outs(sim, images):
    #The activity of the neurons when an image is given as input
    _, acts = nengo.utils.ensemble.tuning_curves(a, sim, inputs=images)
    #The activity multiplied by the weight matrix (calculated in the network) to give the one-hot labels
    return np.dot(acts, sim.data[conn2].weights.T)

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

	
dim =28
#Shift an image
def translate(img,x,y):
    newImg = scipy.ndimage.interpolation.shift(np.reshape(img, (dim,dim), 'F'),(x,y), cval=-1)
    return newImg.T.ravel()
	
#Images to train, starting at random translation
orig_imgs = X_train[:100000].copy()
for img in orig_imgs:
    img[:] = translate(img,random.randint(-6,6),random.randint(-6,6))

#Images translated up a fixed amount from the original random translation
translate_up_imgs = orig_imgs.copy()
for img in translate_up_imgs:
    img[:] = translate(img,0,-1)
    
#Images translated down a fixed amount from the original random translation
translate_down_imgs = orig_imgs.copy()
for img in translate_down_imgs:
    img[:] = translate(img,0,1)
'''
#Images translated right a fixed amount from the original random translation
translate_right_imgs = orig_imgs.copy()
for img in translate_right_imgs:
    img[:] = translate(img,1,0)
    
#Images translated left a fixed amount from the original random translation
translate_left_imgs = orig_imgs.copy()
for img in translate_left_imgs:
    img[:] = translate(img,-1,0)
'''

#Images not used for training, but for testing (all at random translations)
test_imgs = X_test[:1000].copy()
for img in test_imgs:
    img[:] = translate(img,random.randint(-4,4),random.randint(-4,4))
			
with nengo.Simulator(model) as sim:    
    
    #Neuron activities of different mnist images
    #The semantic pointers
    orig_acts = get_activities(sim,orig_imgs)
    translate_up_acts = get_activities(sim,translate_up_imgs)
    translate_down_acts = get_activities(sim,translate_down_imgs)
    '''
	translate_left_acts = get_activities(sim,translate_left_imgs)
    translate_right_acts = get_activities(sim,translate_right_imgs)
    '''
	test_acts = get_activities(sim,test_imgs)
    
    X_test_acts = get_activities(sim,X_test)
    labels_out = get_outs(sim,X_test)
    

    translate_up_after_encoders = get_encoder_outputs(sim,translate_up_imgs)
    translate_down_after_encoders = get_encoder_outputs(sim,translate_down_imgs)
    '''
	translate_left_after_encoders = get_encoder_outputs(sim,translate_left_imgs)
    translate_right_after_encoders = get_encoder_outputs(sim,translate_right_imgs)
    '''
    
    #solvers for a learning rule
    solver_translate_up = nengo.solvers.LstsqL2(reg=1e-8)
    solver_translate_down = nengo.solvers.LstsqL2(reg=1e-8)
    '''
	solver_translate_left = nengo.solvers.LstsqL2(reg=1e-8)
    solver_translate_right = nengo.solvers.LstsqL2(reg=1e-8)
    '''
	solver_word = nengo.solvers.LstsqL2(reg=1e-8)
    solver_translate_up_encoder = nengo.solvers.LstsqL2(reg=1e-8)
    solver_translate_down_encoder = nengo.solvers.LstsqL2(reg=1e-8)
    '''
	solver_translate_left_encoder = nengo.solvers.LstsqL2(reg=1e-8)
    solver_translate_right_encoder = nengo.solvers.LstsqL2(reg=1e-8)
    '''
    
    #find weight matrix between neuron activity of the original image and the translated image
    #weights returns a tuple including information about learning process, just want the weight matrix
    translate_up_weights,_ = solver_translate_up(orig_acts, translate_up_acts)
    translate_down_weights,_ = solver_translate_down(orig_acts, translate_down_acts)
    '''
	translate_left_weights,_ = solver_translate_left(orig_acts, translate_left_acts)
    translate_right_weights,_ = solver_translate_right(orig_acts, translate_right_acts)
    '''
    
    #find weight matrix between labels and neuron activity
    label_weights,_ = solver_word(labels_out,X_test_acts)
    
    
    translate_up_after_encoder_weights,_ = solver_translate_up_encoder(orig_acts,translate_up_after_encoders)
    translate_down_after_encoder_weights,_ = solver_translate_down_encoder(orig_acts,translate_down_after_encoders)
    '''
	translate_left_after_encoder_weights,_ = solver_translate_left_encoder(orig_acts,translate_left_after_encoders)
    translate_right_after_encoder_weights,_ = solver_translate_right_encoder(orig_acts,translate_right_after_encoders)
'''
    
    
#Saving
filename = "activity_to_img_weights_translate" + str(n_hid) +".p"
cPickle.dump(sim.data[conn].weights.T, open( filename, "wb" ) )

filename = "translate_up_weights" + str(n_hid) +".p"
cPickle.dump(translate_up_weights, open( filename, "wb" ) )
filename = "translate_down_weights" + str(n_hid) +".p"
cPickle.dump(translate_down_weights, open( filename, "wb" ) )
'''
filename = "translate_left_weights" + str(n_hid) +".p"
cPickle.dump(translate_left_weights, open( filename, "wb" ) )
filename = "translate_right_weights" + str(n_hid) +".p"
cPickle.dump(translate_right_weights, open( filename, "wb" ) )
'''

filename = "translate_up_after_encoder_weights" + str(n_hid) +".p"
cPickle.dump(translate_up_after_encoder_weights, open( filename, "wb" ) )
filename = "translate_down_after_encoder_weights" + str(n_hid) +".p"
cPickle.dump(translate_down_after_encoder_weights, open( filename, "wb" ) )
'''
filename = "translate_left_after_encoder_weights" + str(n_hid) +".p"
cPickle.dump(translate_left_after_encoder_weights, open( filename, "wb" ) )
filename = "translate_right_after_encoder_weights" + str(n_hid) +".p"
cPickle.dump(translate_right_after_encoder_weights, open( filename, "wb" ) )
'''