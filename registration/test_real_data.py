import SimpleITK as sitk
from skimage import filters, measure, exposure
import numpy as np
from imageview import *
from mayavi import mlab
# import matplotlib.pyplot as plt
from medpy.filter import image
# from scipy.ndimage import binary_fill_holes

def getArray(pathdcm):
    img = sitk.ReadImage(pathdcm)
    voxelData = sitk.GetArrayFromImage(img)
    ConstPixelSpacing = img.GetSpacing()
    return voxelData, ConstPixelSpacing


def main():
    pathdcm = "/Users/Parth/Downloads/1.nii.gz"

    voxelData, spacing = getArray(pathdcm)
    print voxelData.shape
    voxelData /= np.amax(voxelData)
    # voxelData *=255
    # voxelData = np.uint8(voxelData)

    ConstPixelSpacing = (spacing[2], spacing[1], spacing[0])
    thresh = 0
    # voxelData[voxelData<thresh] = 0
    for i in range(voxelData.shape[2]):
        # voxelData[:, :, i] = largest_connected_component(voxelData[:,:,i])
        # print i
        # voxelData[:, :, i] = binary_fill_holes(voxelData[:,:,i])
    # for i in range(voxelData.shape[1]):
        voxelData[:, i] = filters.gaussian(voxelData[:, i],sigma=2)
        thresh = image.otsu(voxelData[:,:,i])
        voxelData[voxelData<thresh] = 0
    # for i in range(voxelData.shape[0]):
        # voxelData[i] = filters.gaussian(voxelData[i],sigma=2)

        # thresh += filters.threshold_mean(voxelData[:, :, i])
        
        # print thresh
        # img = voxelData[:, :, i]
        # img[img < thresh] = 0
        # voxelData[:, :, i] = img
        
    thresh /= voxelData.shape[2]
    # thresh = 0.01
    # voxelData[voxelData<thresh] = 0
    
    # print thresh
    verts, faces, _, _ = measure.marching_cubes_lewiner(
        voxelData, 0, ConstPixelSpacing)
    mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces)
    mlab.show()
    # for i in range(voxelData.shape[2]):
    # voxelData[:,:,i] = exposure.equalize_hist(voxelData[:,:,i])
    # multi_slice_viewer(voxelData)
    # plt.show()


if __name__ == '__main__':
    main()
