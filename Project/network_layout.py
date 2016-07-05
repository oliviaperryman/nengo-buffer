import nengo
import numpy as np
import cPickle
from nengo_extras.data import load_mnist
from nengo_extras.vision import Gabor, Mask
from matplotlib import pylab
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import linalg

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

label_weights = cPickle.load(open("label_weights1000.p", "rb"))
activity_to_img_weights = cPickle.load(open("activity_to_img_weights1000.p", "rb"))
#rotated_clockwise_after_encoder_weights =  cPickle.load(open("rotated_after_encoder_weights_clockwise5000.p", "r"))
rotated_counter_after_encoder_weights =  cPickle.load(open("rotated_after_encoder_weights1000.p", "r"))

#identity_after_encoder_weights = cPickle.load(open("identity_after_encoder_weights1000.p","r"))


#rotation_clockwise_weights = cPickle.load(open("rotation_clockwise_weights1000.p","rb"))
#rotation_counter_weights = cPickle.load(open("rotation_weights1000.p","rb"))

scale_up_after_encoder_weights = cPickle.load(open("scale_up_after_encoder_weights1000.p","r"))
scale_down_after_encoder_weights = cPickle.load(open("scale_down_after_encoder_weights1000.p","r"))
translate_up_after_encoder_weights = cPickle.load(open("translate_up_after_encoder_weights1000.p","r"))
translate_down_after_encoder_weights = cPickle.load(open("translate_down_after_encoder_weights1000.p","r"))
translate_left_after_encoder_weights = cPickle.load(open("translate_left_after_encoder_weights1000.p","r"))
translate_right_after_encoder_weights = cPickle.load(open("translate_right_after_encoder_weights1000.p","r"))




 #A value of zero gives no inhibition

 #A value of zero gives no inhibition

def inhibit_rotate_clockwise(t):
    if t < 1:
        return dim**2
    else:
        return 0
    
def inhibit_rotate_counter(t):
    if t < 1:
        return 0
    else:
        return dim**2
    
def inhibit_identity(t):
    if t < 1:
        return dim**2
    else:
        return dim**2
    
def inhibit_scale_up(t):
    return dim**2
def inhibit_scale_down(t):
    return dim**2

def inhibit_translate_up(t):
    return dim**2
def inhibit_translate_down(t):
    return dim**2
def inhibit_translate_left(t):
    return dim**2
def inhibit_translate_right(t):
    return dim**2

        
def node_func(t,x):
    x[x>0]=1
    x[x<0]=-1
    return x
    
    
def add_manipulation(main_ens,weights,inhibition_func,label):
    #create ensemble for manipulation
    ens_manipulation = nengo.Ensemble(n_hid,dim**2,seed=3,encoders=encoders, label=label,**ens_params)
    #create node for inhibition
    inhib_manipulation = nengo.Node(inhibition_func,label=label)
    #Connect the main ensemble to each manipulation ensemble and back with appropriate transformation
    nengo.Connection(main_ens.neurons, ens_manipulation.neurons, transform = weights.T, synapse=0.1)
    nengo.Connection(ens_manipulation.neurons, main_ens.neurons, transform = weights.T,synapse = 0.1)
    #connect inhibition
    nengo.Connection(inhib_manipulation, ens_manipulation.neurons, transform=[[-1]] * n_hid)
    
    #return ens_manipulation,inhib_manipulation
    
rng = np.random.RandomState(9)
n_hid = 1000
model = nengo.Network(seed=3)
with model:
    #Stimulus only shows for brief period of time
    stim = nengo.Node(lambda t: THREE if t < 0.1 else 0) #nengo.processes.PresentInput(labels,1))#
    
    ens_params = dict(
        eval_points=X_train,
        neuron_type=nengo.LIF(), #Why not use LIF?
        intercepts=nengo.dists.Choice([-0.5]),
        max_rates=nengo.dists.Choice([100]),
        )
        
    
    # linear filter used for edge detection as encoders, more plausible for human visual system
    encoders = Gabor().generate(n_hid, (11, 11), rng=rng)
    encoders = Mask((28, 28)).populate(encoders, rng=rng, flatten=True)


    #Ensemble that represents the image with different transformations applied to it
    ens = nengo.Ensemble(n_hid, dim**2, seed=3, encoders=encoders, **ens_params)
    

    #Connect stimulus to ensemble, transform using learned weight matrices
    nengo.Connection(stim, ens, transform = np.dot(label_weights,activity_to_img_weights).T)
    
    '''
    #Recurrent connection on the neurons of the ensemble to perform the rotation
    #nengo.Connection(ens.neurons, ens.neurons, transform = rotated_counter_after_encoder_weights.T, synapse=0.1)      
    #nengo.Connection(ens.neurons, ens.neurons, transform = low_pass_weights.T, synapse=0.1)      

    #Identity ensemble
    #ens_iden = nengo.Ensemble(n_hid,dim**2, seed=3, encoders=encoders, **ens_params)
    #Rotation ensembles
    ens_clock_rot = nengo.Ensemble(n_hid,dim**2,seed=3,encoders=encoders, **ens_params)
    ens_counter_rot = nengo.Ensemble(n_hid,dim**2,seed=3,encoders=encoders, **ens_params)
    
    
    #Inhibition nodes
    #inhib_iden = nengo.Node(inhibit_identity)
    inhib_clock_rot = nengo.Node(inhibit_rotate_clockwise)
    inhib_counter_rot = nengo.Node(inhibit_rotate_counter)

    #Connect the main ensemble to each manipulation ensemble and back with appropriate transformation
    #Identity
    #nengo.Connection(ens.neurons, ens_iden.neurons,transform=identity_after_encoder_weights.T,synapse=0.1)
    #nengo.Connection(ens_iden.neurons, ens.neurons,transform=identity_after_encoder_weights.T,synapse=0.1)
    #Clockwise
    nengo.Connection(ens.neurons, ens_clock_rot.neurons, transform = rotated_counter_after_encoder_weights.T,synapse=0.1)
    nengo.Connection(ens_clock_rot.neurons, ens.neurons, transform = rotated_counter_after_encoder_weights.T,synapse = 0.1)
    #Counter-clockwise
    nengo.Connection(ens.neurons, ens_counter_rot.neurons, transform = rotated_counter_after_encoder_weights.T, synapse=0.1)
    nengo.Connection(ens_counter_rot.neurons, ens.neurons, transform = rotated_counter_after_encoder_weights.T, synapse=0.1)
  
    #nengo.Connection(ens.neurons, ens_counter_rot.neurons, transform = low_pass_weights.T, synapse=0.1)
    #nengo.Connection(ens_counter_rot.neurons, ens.neurons, transform = low_pass_weights.T, synapse=0.1)

    #n = nengo.Node(node_func, size_in=dim**2)
    #nengo.Connection(ens.neurons,n,transform=activity_to_img_weights.T)#, synapse=0.1)
    #nengo.Connection(n,ens_counter_rot)#,synapse=0.1)

    #Connect the inhibition nodes to each manipulation ensemble
    #nengo.Connection(inhib_iden, ens_iden.neurons, transform=[[-1]] * n_hid)
    nengo.Connection(inhib_clock_rot, ens_clock_rot.neurons, transform=[[-1]] * n_hid)
    nengo.Connection(inhib_counter_rot, ens_counter_rot.neurons, transform=[[-1]] * n_hid)

    '''
    
        
    add_manipulation(ens,rotated_counter_after_encoder_weights, inhibit_rotate_clockwise,"clockwise")
    add_manipulation(ens,rotated_counter_after_encoder_weights, inhibit_rotate_counter,"counter")
    add_manipulation(ens,scale_up_after_encoder_weights, inhibit_scale_up,"scale up")
    add_manipulation(ens,scale_down_after_encoder_weights, inhibit_scale_down, "scale down")
    add_manipulation(ens,translate_up_after_encoder_weights, inhibit_translate_up,"up")
    add_manipulation(ens,translate_down_after_encoder_weights, inhibit_translate_down,"down")
    add_manipulation(ens,translate_left_after_encoder_weights, inhibit_translate_left,"left")
    add_manipulation(ens,translate_right_after_encoder_weights, inhibit_translate_right,"right")
    
    

    
    #Collect output, use synapse for smoothing
    probe = nengo.Probe(ens.neurons,synapse=0.1)
    