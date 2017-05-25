import numpy as np
import matplotlib.pyplot as plt

def read_rates(root, net, size, fname='rate.bin', dtype=np.float32):
    r = np.memmap(root + '/' + net + '/' + fname, dtype)
    r = r[:int(r.size/size) * size]
    return r.reshape(int(r.size/size), size).T

def read_activations(root, net_post, net_pre, size, dtype=np.float32):
    return read_rates(root, net_post, size,
                      'activation_' + net_pre + '_' + net_post + '.bin', dtype)

def accumulate_activation(root, net_post, nets_pre, size, sl=None, dtype=np.float32):
    activs = [read_activations(root, net_post, net_pre, size, dtype)
              for net_pre in nets_pre]
    min_len = min([a.shape[1] for a in activs])
    activs = [a[:, :min_len] for a in activs]
    if sl is None:
        return np.sum(activs, 0)
    else:
        return np.sum([a[:, sl] for a in activs], 0)

def read_weights(root, net_post, net_pre,
                 size_post, size_pre, dtype=np.float32):
    w = np.memmap(root + '/' + net_post + '/'
                    + 'weights_' + net_pre + '_' + net_post + '.bin', dtype)
    w = w[:int(w.size/(size_pre*size_post)) * size_pre * size_post]
    w = w.reshape(int(w.size/(size_pre*size_post)), size_pre, size_post)
    return w

def imshow(arr):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    fig, ax = plt.subplots()
    w, h = plt.figaspect(arr.shape[0] / (arr.shape[1]+10))
    fig.set_size_inches(w, h)
    im = ax.imshow(arr, interpolation='none', aspect='auto')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)

    fig.colorbar(im, cax=cax, orientation='vertical')

    return fig, ax
