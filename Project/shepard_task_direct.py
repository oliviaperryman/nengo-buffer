import nengo
from nengo_extras.data import load_mnist
import numpy as np
import scipy.ndimage
from skimage.measure import compare_ssim as ssim

input_shape = (1,28,28)


def display_func(t, x, input_shape=input_shape):
    import base64
    import PIL
    import cStringIO
    from PIL import Image

    values = x.reshape(input_shape)
    values = values.transpose((1, 2, 0))
    vmin, vmax = values.min(), values.max()
    values = (values - vmin) / (vmax - vmin + 1e-8) * 255.
    values = values.astype('uint8')

    if values.shape[-1] == 1:
        values = values[:, :, 0]

    png = PIL.Image.fromarray(values)
    buffer = cStringIO.StringIO()
    png.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())

    display_func._nengo_html_ = '''
        <svg width="100%%" height="100%%" viewbox="0 0 100 100">
        <image width="100%%" height="100%%"
               xlink:href="data:image/png;base64,%s"
               style="image-rendering: pixelated;">
        </svg>''' % (''.join(img_str))


#Load mnist images
(X_train, y_train), (X_test, y_test) = load_mnist()

'''
def intense(img):
    newImg = img.copy()
    #newImg = scipy.ndimage.gaussian_filter(newImg, sigma=1)
    newImg[newImg < 0.01] = -1
    newImg[newImg > 0.01] = 1
    return newImg
'''

#Rotate after 0.1 seconds
def rotate_func(t,x):
    if(t<0.1):
        return x
    return scipy.ndimage.interpolation.rotate(x.reshape(28,28),-1,reshape=False).ravel()

'''
def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err
'''

#Calculare similarity between two images
def similarity(x):
    return ssim(x[:784].reshape(28,28),x[784:].reshape(28,28))
    #return mse(x[:784].reshape(28,28),x[784:].reshape(28,28))

#Threshold for similarity
def answer_func(x):
    return 1 if x> 0.45 else 0


img =X_train[7] #works past 360
rot_img =scipy.ndimage.interpolation.rotate(img.reshape(28,28),90).ravel()

model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: img if t<0.1 else 0)
    stim_rot = nengo.Node(lambda t: rot_img if t <0.1 else 0)
    a = nengo.Ensemble(n_neurons=50, dimensions=784, neuron_type=nengo.Direct())
    nengo.Connection(stim, a)
    nengo.Connection(a,a)
    a_rot = nengo.Ensemble(n_neurons=50, dimensions=784, neuron_type=nengo.Direct())
    nengo.Connection(stim_rot, a_rot)
    rot_node = nengo.Node(rotate_func,size_in = a_rot.size_out)
    nengo.Connection(a_rot,rot_node)
    nengo.Connection(rot_node,a_rot)

    display_node = nengo.Node(display_func, size_in=a.size_out)
    nengo.Connection(a, display_node, synapse=None)
    
    display_node_rot = nengo.Node(display_func, size_in=a_rot.size_out)
    nengo.Connection(a_rot, display_node_rot, synapse=None)
    
    
    combine = nengo.Ensemble(50, 784*2, neuron_type=nengo.Direct())
    
    nengo.Connection(a,combine[:784])
    nengo.Connection(a_rot, combine[784:])

    
    result = nengo.Node(lambda t,x: x, size_in=1)
    
    nengo.Connection(combine, result, function = similarity)
    
    answer = nengo.Node(None,size_in=1)
    nengo.Connection(result,answer, function=answer_func)
    
    