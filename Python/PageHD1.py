import numpy as np
import matplotlib.pyplot as plt

def read_rates(root, net, size, fname='rate.bin', dtype=np.float32):
    r = np.memmap(root + '/' + net + '/' + fname, dtype)
    return r.reshape(int(r.size/size), size).T

def read_activations(root, net_post, net_pre, size, dtype=np.float32):
    return read_rates(root, net_post, size,
                      'activation_' + net_pre + '_' + net_post + '.bin', dtype)

def accumulate_activation(root, net_post, nets_pre, size, dtype=np.float32):
    return np.sum([read_activations(root, net_post, net_pre, size, dtype)
                   for net_pre in nets_pre], 0)

def read_weights(root, net_post, net_pre,
                 size_post, size_pre, dtype=np.float32):
    w = np.memmap(root + '/' + net_post + '/'
                    + 'weights_' + net_pre + '_' + net_post + '.bin', dtype)
    w = w.reshape(w.size/(size_pre*size_post), size_pre, size_post)
    return w.transpose((0, 2, 1))

def imshow(arr):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    fig, ax = plt.subplots()
    im = ax.imshow(arr, interpolation='none', aspect='auto')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)

    fig.colorbar(im, cax=cax, orientation='vertical')

    return fig, ax
