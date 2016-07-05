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
#Scale an image
def scale(img, scale):
    newImg = scipy.ndimage.interpolation.zoom(np.reshape(img, (dim,dim), 'F').T,scale,cval=-1)
    #If its scaled up
    if(scale >1):
        newImg = newImg[len(newImg)/2-(dim/2):-(len(newImg)/2-(dim/2)),len(newImg)/2-(dim/2):-(len(newImg)/2-(dim/2))]
        if len(newImg) >28:
            newImg = newImg[:28,:28]
        newImg = newImg.ravel()
    else: #Scaled down
        m = np.zeros((dim,dim))
        m.fill(-1)
        m[(dim-len(newImg))/2:(dim-len(newImg))/2+len(newImg),(dim-len(newImg))/2:(dim-len(newImg))/2+len(newImg)] = newImg
        newImg = m
    return newImg.ravel()
	
#Images to train, starting at random size
orig_imgs = X_train[:100000].copy()
for img in orig_imgs:
    while True:
        try:
            img[:] = scale(img,random.uniform(0.5,1.5))
            break
        except:
            img[:] = img

#Images scaled up a fixed amount from the original random scaling
scaled_up_imgs = orig_imgs.copy()
for img in scaled_up_imgs:
    img[:] = scale(img,1.1)
    
#Images scaled down a fixed amount from the original random scaling
scaled_down_imgs = orig_imgs.copy()
for img in scaled_down_imgs:
    img[:] = scale(img,0.9)
    
#Images not used for training, but for testing (all at random orientations)
test_imgs = X_test[:1000].copy()
for img in test_imgs:
    img[:] = scipy.ndimage.interpolation.rotate(np.reshape(img,(28,28)),
                                                (np.random.randint(360)),reshape=False,mode="nearest").ravel()

#Images not used for training, but for testing (all at random sizes)
test_imgs = X_test[:1000].copy()
for img in test_imgs:
    while True:
        try:
            img[:] = scale(img,random.uniform(0.5,1.5))
            break
        except:
            img[:] = img
			
with nengo.Simulator(model) as sim:    
    
    #Neuron activities of different mnist images
    #The semantic pointers
    orig_acts = get_activities(sim,orig_imgs)
    scaled_up_acts = get_activities(sim,scaled_up_imgs)
    scaled_down_acts = get_activities(sim,scaled_down_imgs)
    test_acts = get_activities(sim,test_imgs)
    
    X_test_acts = get_activities(sim,X_test)
    labels_out = get_outs(sim,X_test)
    
    scaled_up_after_encoders = get_encoder_outputs(sim,scaled_up_imgs)
    scaled_down_after_encoders = get_encoder_outputs(sim,scaled_down_imgs)
    
    
    #solvers for a learning rule
    solver_scale_up = nengo.solvers.LstsqL2(reg=1e-8)
    solver_scale_down = nengo.solvers.LstsqL2(reg=1e-8)
    solver_word = nengo.solvers.LstsqL2(reg=1e-8)
    solver_scaled_up_encoder = nengo.solvers.LstsqL2(reg=1e-8)
    solver_scaled_down_encoder = nengo.solvers.LstsqL2(reg=1e-8)
    
    
    
    #find weight matrix between neuron activity of the original image and the rotated image
    #weights returns a tuple including information about learning process, just want the weight matrix
    scale_up_weights,_ = solver_scale_up(orig_acts, scaled_up_acts)
    scale_down_weights,_ = solver_scale_down(orig_acts, scaled_down_acts)
    
    
    #find weight matrix between labels and neuron activity
    label_weights,_ = solver_word(labels_out,X_test_acts)
    
    
    scaled_up_after_encoder_weights,_ = solver_scaled_up_encoder(orig_acts,scaled_up_after_encoders)
    scaled_down_after_encoder_weights,_ = solver_scaled_down_encoder(orig_acts,scaled_down_after_encoders)
    
#Saving
filename = "activity_to_img_weights_scale" + str(n_hid) +".p"
cPickle.dump(sim.data[conn].weights.T, open( filename, "wb" ) )

filename = "scale_up_weights" + str(n_hid) +".p"
cPickle.dump(scale_up_weights, open( filename, "wb" ) )
filename = "scale_down_weights" + str(n_hid) +".p"
cPickle.dump(scale_down_weights, open( filename, "wb" ) )

filename = "scale_up_after_encoder_weights" + str(n_hid) +".p"
cPickle.dump(scaled_up_after_encoder_weights, open( filename, "wb" ) )
filename = "scale_down_after_encoder_weights" + str(n_hid) +".p"
cPickle.dump(scaled_down_after_encoder_weights, open( filename, "wb" ) )
