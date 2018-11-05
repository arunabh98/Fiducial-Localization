import dicom_numpy
import dicom
import os
from natsort import natsorted
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools
import numpy as np
import scipy.ndimage as snd
import mudicom
from mayavi import mlab
import time
from skimage import measure
import numpy as np
from sklearn.neighbors import NearestNeighbors

def getSurfaceVoxels(voxelData):
	
	andInX = np.logical_and.reduce((voxelData[:, 0:voxelData.shape[1] - 2, :],
								voxelData[:, 1:voxelData.shape[1] - 1, :],
								voxelData[:, 2:voxelData.shape[1], :]))
	andInXY = np.logical_and.reduce((andInX[0:andInX.shape[0] - 2, :, :],
								andInX[1:andInX.shape[0] - 1, :, :],
								andInX[2:andInX.shape[0], :, :]))
	andInXYZ = np.logical_and.reduce((andInXY[:, :, 0:andInXY.shape[2] - 2],
								andInXY[:, :, 1:andInXY.shape[2] - 1],
								andInXY[:, :, 2:andInXY.shape[2]]))

	voxelFilteredData = np.logical_and(np.logical_not(andInXYZ),
						voxelData[1:voxelData.shape[0] - 1,
									  1:voxelData.shape[1] - 1,
									  1:voxelData.shape[2] - 1])

	onVoxelsX, onVoxelsY, onVoxelsZ = np.nonzero(voxelFilteredData == 1)
	onVoxels = np.stack((onVoxelsX, onVoxelsY, onVoxelsZ), axis=1)

	# Offset the indices
	surfaceVoxels = onVoxels + 1


	return surfaceVoxels



def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    ##assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    ## assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    ## assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i




def readDicomData(path):
	lstFilesDCM = []
	# may want to exclude the first dicom image in some files
	for root, directory, fileList in os.walk(path):
		for filename in fileList:
			if filename==".DS_Store":
				continue
			lstFilesDCM.append(filename)
	''' 
	the function natsorted() from natsort library does natural sorting
	i.e the files are in the order "IM1,IM2,IM3..." 
	instead of "IM1,IM10,IM100.." which is the lexicographical order
	'''
	lstFilesDCM = natsorted(lstFilesDCM)
	data = [dicom.read_file(path + '/' + f) for f in lstFilesDCM]
	return data

def get3DRecon(data):
	RefDs = data[0]
	ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

	try:
	    voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(data)
	except dicom_numpy.DicomImportException as e:
	    # invalid DICOM data
	    raise NameError('Unable to do 3D reconstruction. Slice missing? or incompatible slice data?')
	# arbitrary for now, can be set to different values for CT scan as in Hounsfield unit,
	# bone is from +700 to +3000
	"""
	
	upper_thresh = 0
	lower_thresh = 0

	voxel_ndarray[voxel_ndarray > upper_thresh] = 1
	voxel_ndarray[voxel_ndarray <= lower_thresh] = 0
	
	"""

	return (voxel_ndarray,ijk_to_xyz)

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


def view_pointcloud(surfaceVoxels,color):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(surfaceVoxels[:, 0], surfaceVoxels[:, 1], surfaceVoxels[:, 2],color=color)
	

directory_ref = "/home/j_69/Fiducial Localization - MRI Scans/glass/Glass Scan Axial 1.25 mm/DICOM/PA1/ST1/SE2"
data_ref = readDicomData(directory_ref)
voxel_ndarray_ref = get3DRecon(data_ref)[0]
ijk_to_xyz_ref = get3DRecon(data_ref)[1]
surfaceVoxels_ref = getSurfaceVoxels(voxel_ndarray_ref)
# surfaceVoxels = np.asarray(surfaceVoxels)
print surfaceVoxels_ref

view_pointcloud(surfaceVoxels_ref,color='y')
plt.show()

##multi_slice_viewer(voxel_ndarray_ref)
##plt.show()


 ### Using Rawfloating image, as patient data missing for this view

directory_flo1 = "/home/j_69/Fiducial Localization - MRI Scans/glass/Glass Scan Coronal 0.9 mm/DICOM/PA1/ST1/SE2" ## edit this accordingly
slices = 256 ## edit this accordingly

mu = []
arry = []
for i in range(1,slices):
    mu.append(mudicom.load(directory_flo1+"/IM"+str(i)))
    arry.append(mu[i-1].image.numpy)
voxel_ndarray_flo1 = np.array(arry)

surfaceVoxels_flo1 = getSurfaceVoxels(voxel_ndarray_flo1)
print surfaceVoxels_flo1


view_pointcloud(surfaceVoxels_ref,color='b')
plt.show()

print("Applying ICP...")

ijkmat = icp(surfaceVoxels_ref,surfaceVoxels_flo1, max_iterations = 40)
print(ijkmat)



"""
voxel_ndarray_flo1 = snd.rotate(voxel_ndarray_ref,15)

print("Transformation performed on reference image")

surfaceVoxels_flo1 = getSurfaceVoxels(voxel_ndarray_flo1)
# surfaceVoxels = np.asarray(surfaceVoxels)
arr = np.delete(surfaceVoxels_flo1,1,0)
print surfaceVoxels_flo1


##multi_slice_viewer(voxel_ndarray_flo1)
##plt.show()
"""


"""
directory_flo1 = "/home/j_69/Fiducial Localization - MRI Scans/glass/Glass Scan Sagittal 0.9 mm/DICOM/PA1/ST1/SE2"
data_flo1 = readDicomData(directory_flo1)
voxel_ndarray_flo1 = get3DRecon(data_flo1)[0]
ijk_to_xyz_flo1 = get3DRecon(data_flo1)[1]
surfaceVoxels_flo1 = getSurfaceVoxels(voxel_ndarray_flo1)
# surfaceVoxels = np.asarray(surfaceVoxels)
print surfaceVoxels_flo1
print(ijk_to_xyz_flo1)
"""





"""
f = open(directory+"/reconstructed.txt",'w')


for i in range(len(voxel_ndarray)):
	for j in range(len(voxel_ndarray[i])):
		for k in range(len(voxel_ndarray[i][j])):
			f.write(str(voxel_ndarray[i][j][k]))
			f.write(" ")
		f.write(",")
	f.write("\n")

f.close()
"""
## multi_slice_viewer(voxel_ndarray)
##plt.show()
