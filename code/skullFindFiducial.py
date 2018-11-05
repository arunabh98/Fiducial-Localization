# -*- coding: utf-8 -*-
"""
Autonomous localization of fiducial markers for IGNS.
This script contains utilities for extracting, registering
and evaluating the surfel patches for fiducial markers.

Authors: P. Khirwadkar, H. Loya, D. Shah, R. Chaudhry,
A. Ghosh & S. Goel (For Inter IIT Technical Meet 2018)
Copyright © 2018 Indian Institute of Technology, Bombay
"""
import numpy as np
from scipy.spatial import cKDTree, distance
from skimage import measure
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MeanShift
from mayavi import mlab
from skullReconstruct import *
from genFiducialPC import *
import copy
import time
from joblib import Parallel, delayed


vFiducial = np.array([])
fFiducial = np.array([])
ConstPixelSpacing = (1.0, 1.0, 1.0)
c = 0


def getNeighborVoxel(pointCloud, points, r):
    """
    Computes the nearest neighbour to each point in points, from the
    pointCloud passed.
    """
    kdt = cKDTree(pointCloud)
    neighbor = kdt.query_ball_point(points, r)
    return neighbor


def nearest_neighbor(src, dst):
    """
    Finds the nearest (Euclidean) neighbor in dst for each point in src
    Input:
            src: Nxm array of points
            dst: Nxm array of points
    Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
    """
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding
    points A to B in dim spatial dimensions.
    Input:
      A: N*dim numpy array of corresponding points
      B: N*dim numpy array of corresponding points
    Returns:
      T: (dim+1)x(dim+1) homogeneous transformation matrix that maps A on to B
      R: dim*dim rotation matrix
      t: dim*1 translation vector
    """
    dim = A.shape[1]

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
        Vt[dim - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(dim + 1)
    T[:dim, :dim] = R
    T[:dim, dim] = t

    return T, R, t


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps
    points in A to points B
    Input:
            A: N*dim numpy array of source mD points
            B: N*dim numpy array of destination mD point
            init_pose: (dim+1)*(dim+1) homogeneous transformation
            max_iterations: exit algorithm after max_iterations
            tolerance: convergence criteria
    Output:
            T: final homogeneous transformation that maps A on to B
            distances: Euclidean distances (errors) of the nearest neighbor
            i: number of iterations to converge
    """
    dim = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((dim + 1, A.shape[0]))
    dst = np.ones((dim + 1, B.shape[0]))
    src[:dim, :] = np.copy(A.T)
    dst[:dim, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    # display the point cloud, after initial transformation

    prev_error = 0
    error_arr = []
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination
        # points
        distances, indices = nearest_neighbor(src[:dim, :].T, dst[:dim, :].T)

        # compute the transformation between the current source and nearest
        # destination points
        T, _, _ = best_fit_transform(src[:dim, :].T, dst[:dim, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        error_arr.append(mean_error)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:dim, :].T)
    min_error = np.array(error_arr).min()

    return T, min_error / len(distances), i


def apply_affine(A, init_pose):
    """
    Applies the specified affine transform to matrix A.
    """
    dim = A.shape[1]
    src = np.ones((dim + 1, A.shape[0]))
    src[:dim, :] = np.copy(A.T)
    if init_pose is not None:
        src = np.dot(init_pose, src)
    return src[:dim, :].T


def find_init_transfo(evec1, evec2):
    """
    @
    """
    e_cross = np.cross(evec1, evec2)
    e_cross1 = e_cross[0]
    e_cross2 = e_cross[1]
    e_cross3 = e_cross[2]
    i = np.identity(3)
    v = np.zeros((3, 3))
    v[1, 0] = e_cross3
    v[2, 0] = -e_cross2
    v[0, 1] = -e_cross3
    v[2, 1] = e_cross1
    v[0, 2] = e_cross2
    v[1, 2] = -e_cross1
    v2 = np.dot(v, v)
    c = np.dot(evec1, evec2)
    # will not work in case angle of rotation is exactly 180 degrees
    if(c == -1):
        R = -1 * np.eye(3)
    else:
        R = i + v + (v2 / (1 + c))

    T = np.identity(4)
    T[0:3, 0:3] = R
    T = np.transpose(T)
    R = np.transpose(R)
    #com = [img.resolution[0]*len(img)/2,img.resolution[1]*len(img[0])/2,img.resolution[2]*len(img[0][0])/2]
    [tx, ty, tz] = [0, 0, 0]
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def genPatch(surfaceVoxelCoord, normals, point, neighbor, PixelSpacing):
    """
    Generates a patch around each surfel provided, along with a modified patch,
    with its surface normal aligned to the vertical [0 0 1].
    """
    patch = surfaceVoxelCoord[neighbor]
    try:
        neigh = NearestNeighbors(n_neighbors=4)
        neigh.fit(patch)
        distances, indices = neigh.kneighbors(
            point.reshape(1, -1), return_distance=True)
        center = point
        orignalPatch = copy.deepcopy(patch)
        patch -= center
        norm = normals[neighbor[indices]].reshape(-1, 3)
        norm = np.sum(norm, axis=0) / 4
        norm = norm.reshape(1, -1)

        T = find_init_transfo(
            np.array([0.0, 0.0, 1.0]), copy.deepcopy(norm[0]))
        alignedPatch = apply_affine(copy.deepcopy(patch), T)
        alignedNorm = apply_affine(copy.deepcopy(norm), T)
    except:
        alignedPatch = np.array([1])
        alignedNorm = np.array([[1.0, 0.0, 0.0]])
        orignalPatch = np.array([1])
        print("exceptions")
    return alignedPatch, alignedNorm[0], orignalPatch


def genFiducialModel(PixelSpacing):
    """
    Generates a model of the Fiducial marker, based on known geometry.
    This routine returns vertices, faces and normals of the model,
    as returned by the Marching Cubes algorithm.
    """
    global ConstPixelSpacing
    ConstPixelSpacing = PixelSpacing
    innerD = 4  # in mm
    outerD = 14 * ConstPixelSpacing[0]  # in mm
    height = 2  # in mm
    mPixel = np.uint8(np.round(outerD / ConstPixelSpacing[0]))
    if mPixel % 2 != 0:
        mPixel += 16
    else:
        mPixel += 15
    mLayer = height / ConstPixelSpacing[2]

    fiducial = np.zeros((mPixel, mPixel, int(mLayer) + 2))
    for l in range(fiducial.shape[2]):
        for i in range(mPixel):
            for j in range(mPixel):
                d = np.sqrt(((i - (mPixel - 1) * 0.5) * ConstPixelSpacing[0])**2 +
                            ((j - (mPixel - 1) * 0.5) * ConstPixelSpacing[1])**2)
                if d <= outerD * 0.5 and d >= innerD * 0.5 and l <= mLayer:
                    fiducial[i, j, l] = 1
                elif d > (outerD * 0.5) and d < ((outerD * 0.5) + 1) and l <= mLayer:
                    fiducial[i, j, l] = 1 - (d - (outerD * 0.5))
                elif d < innerD * 0.5 and d < ((innerD * 0.5) - 1) and l <= mLayer:
                    fiducial[i, j, l] = 1 + (d - (innerD * 0.5))
    disk = np.zeros((fiducial.shape[0], fiducial.shape[1]))
    for i in range(fiducial.shape[0]):
        for j in range(fiducial.shape[1]):
            d = np.sqrt(((i - (mPixel - 1) * 0.5) * ConstPixelSpacing[0])**2 +
                        ((j - (mPixel - 1) * 0.5) * ConstPixelSpacing[1])**2)
            if d <= innerD * 0.5:
                disk[i, j] = 1
    x, y = np.where(disk == 1)
    z = np.zeros(x.size)
    x = np.float64(x) * ConstPixelSpacing[0]
    y = np.float64(y) * ConstPixelSpacing[1]
    x -= np.sum(x) / x.size
    y -= np.sum(y) / y.size
    vert = np.stack([x, y, z], axis=1)

    vertFiducial, fFiducial, nFiducial, valFiducial = measure.marching_cubes_lewiner(
        fiducial, 0, ConstPixelSpacing)

    # mlab.triangular_mesh([vert[0] for vert in vertFiducial],
    #               [vert[1] for vert in vertFiducial],
    #               [vert[2] for vert in vertFiducial], fFiducial)
    # mlab.show()

    # assert False, "Stop"

    vertFiducial = vertFiducial - np.sum(
        vertFiducial[vertFiducial[:, 2] <= 0],
        axis=0) / vertFiducial[vertFiducial[:, 2] <= 0].shape[0]
    vertFiducial = np.append(vertFiducial, vert, axis=0)
    return vertFiducial, fFiducial, nFiducial


def filterFiducials(cost, points, numMarkers):
    points = np.float64(copy.deepcopy(points)) * ConstPixelSpacing
    cost = np.array(cost)
    bestIndices = np.argsort(cost)
    bestIndices = bestIndices[:numMarkers]
    # for i in range(bestIndices.shape[0]):
    #   print type(bestIndices[i])
    # print bestIndices
    # print bestIndices.shape
    # print cost.shape
    # print np.max(bestIndices)
    bestCosts = cost[bestIndices]
    # points = points[bestIndices,:]
    # bestCosts = bestCosts[:numMarkers]
    # print bestIndices.shape
    bestPoints = points[bestIndices, :]
    # print bestPoints.shape

    filteredIndices = []

    # Perform Mean Shift Clustering
    meanShift = MeanShift(bandwidth=10)
    # params = meanShift.get_params()
    # bw = params['bandwidth']
    # meanShift.set_params(bandwidth=bw/10.0)
    meanShift.fit(bestPoints)
    labels = meanShift.labels_
    nClusters = len(np.unique(labels))
    print nClusters
    for i in range(nClusters):
        costsPerFiducial = bestCosts[labels == i]
        print costsPerFiducial.shape
        indicesPerFiducial = bestIndices[labels == i]
        filteredIndices.append(indicesPerFiducial[np.argmin(costsPerFiducial)])

    return filteredIndices

sigma = 100.0  # A large value
points_thresh = 400  # A threshold on number of points in a typical patch
def checkFiducial(pointCloud, poi, normalstotal, PixelSpacing):
    """
    This routine performs template matching between a patch around each
    point of interest (from the point cloud), and a known Fiducial marker
    model. It returns the cost (normalised distance from ICP) and the patches.
    """
    global vFiducial, fFiducial, ConstPixelSpacing

    global globalPointCloud
    global globalNormalstotal
    globalPointCloud = pointCloud
    globalNormalstotal = normalstotal

    start_time = time.time()
    ConstPixelSpacing = PixelSpacing

    if vFiducial.size == 0:
        vFiducial = genFiducialPC()
    alignedPatches = []
    patches = []
    point = np.float64(copy.deepcopy(poi)) * ConstPixelSpacing
    neighbor1 = getNeighborVoxel(pointCloud, point, r=7.75)
    neighbor1 = np.array(neighbor1)

    # results = list(Parallel(n_jobs=2)(delayed(computePointCloudDistance)(pnt, n_pnt) for pnt, n_pnt in zip(point, neighbor1)))
    # cost = np.array([result[0] for result in results], dtype=np.float64)
    # count = np.count_nonzero([result[1] for result in results])

    cost = np.array([], dtype=np.float64)
    count = 0
    sigma = 100.0  # A large value
    points_thresh = 400  # A threshold on number of points in a typical patch
    for i in range(len(point)):
        alignedPatch, _, _ = genPatch(pointCloud, normalstotal, point[
            i], np.array(neighbor1[i]).astype(int), ConstPixelSpacing)
        if(len(alignedPatch) > points_thresh):
            cost = np.append(
                cost, icp(alignedPatch, vFiducial, max_iterations=1)[1])
        else:
            count += 1
            cost = np.append(cost, sigma)
        # alignedPatches.append(aligP)
        # patches.append(P)
    # patches = np.array(patches)  # Orignal patch
    # alignedPatches = np.array(alignedPatches)  # Transformed patch

    # cost = []
    # count = 0
    # sigma = 100  # A large value
    # points_thresh = 400  # A threshold on number of points in a typical patch
    # for i in range(len(point)):
    #     if(len(alignedPatches[i]) > points_thresh):
    #         cost.append(icp(alignedPatches[i], vFiducial, max_iterations=1)[1])
    #     else:
    #         count += 1
    #         cost.append(sigma)
    print("ICP Completed!" + str(count) + " of small point clouds detected!")

    return cost, neighbor1  # , patches


def computePointCloudDistance(point, neighbor_pnt):
    cost = sigma
    count = 0
    alignedPatch, _, _ = genPatch(globalPointCloud, globalNormalstotal, point,
        np.array(neighbor_pnt).astype(int), ConstPixelSpacing)
    if(len(alignedPatch) > points_thresh):
        cost = icp(alignedPatch, vFiducial, max_iterations=1)[1]
    else:
        count = 1

    return cost, count


def visualiseFiducials(cost, neighbourIndices, points, pointCloud, verts, faces, num_markers=100, show_skull=True, show_markers=True):
    """
    Collects the top __ fiducial markers, filters them using the Mean Shift algorithm,
    and then renders with the original 3D scan, on Mayavi for
    visualisation and verification.
    """
    indices = np.array(filterFiducials(cost, points, num_markers))
    # print neighbourIndices.shape
    # print indices.shape
    # fiducialNeighbourIndices = neighbourIndices[indices,:]
    # print fiducialNeighbourIndices.shape
    # fiducialNeighbours = pointCloud[fiducialNeighbourIndices,:]
    # print fiducialNeighbours.shape
    print len(indices)
    # cost_sorted = np.sort(cost)
    colormap = np.random.rand(100, 3)
    if show_skull:
        mlab.triangular_mesh([vert[0] for vert in verts],
                             [vert[1] for vert in verts],
                             [vert[2] for vert in verts], faces)
    if show_markers:

        # Plot the top __ markers!
        for i in range(len(indices)):
            # print i
            # TODO - Find Patch now
            fiducialNeighbourIndices = neighbourIndices[indices[i]]
            patch = pointCloud[fiducialNeighbourIndices]
            mlab.points3d(patch[:, 0], patch[:, 1], patch[
                :, 2], color=tuple(colormap[i]))

    mlab.show()
