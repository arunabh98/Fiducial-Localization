"""

Image Viewer
Press 'j' and 'k' to navigate through slices
Mention the directory name in fpath
Mention the number of slices in slices

"""
import matplotlib.pyplot as plt
import mudicom as dicom
import numpy as np

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)
                
def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


fpath = "/home/j_69/Fiducial Localization - MRI Scans/pvc/Sequential Scan/DICOM/PA1/ST1/SE2" ## edit this accordingly
slices = 176 ## edit this accordingly

mu = []
arry = []
for i in range(1,slices):
    mu.append(dicom.load(fpath+"/IM"+str(i)))
    arry.append(mu[i-1].image.numpy)


multi_slice_viewer(np.array(arry))
plt.show()

