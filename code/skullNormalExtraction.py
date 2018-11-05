# -*- coding: utf-8 -*-
"""
Autonomous localization of fiducial markers for IGNS.
This script contains utilities for surface and normal
handling, for 3D data.

Authors: P. Khirwadkar, H. Loya, D. Shah, R. Chaudhry,
A. Ghosh & S. Goel (For Inter IIT Technical Meet 2018)
Copyright © 2018 Indian Institute of Technology, Bombay
"""
import numpy as np
from skimage import measure
from sklearn.neighbors import NearestNeighbors


def getSurfaceVoxels(voxelData):
    """
    Processes the voxel grid to identify surface voxels.
    """
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


def findSurfaceNormals(surfaceVoxels, voxelData, ConstPixelSpacing):
    """
    Uses surface voxels and voxel grid data to compute all outward surface
    normals by manipulating the nearest neighbours' normals.

    """
    verts, normals, faces = getSurfaceMesh(voxelData, ConstPixelSpacing)

    surfaceVoxels = np.float64(surfaceVoxels) * ConstPixelSpacing

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(verts)
    distances, indices = nbrs.kneighbors(surfaceVoxels)
    surfaceNormals = normals[indices[:]]
    surfaceNormals = surfaceNormals.reshape(
        surfaceNormals.shape[0], surfaceNormals.shape[2])

    surfaceNormals_out, surfaceVoxels_out = getOutwardNormals(
        surfaceNormals, surfaceVoxels)

    return surfaceNormals_out, surfaceVoxels_out, verts, faces


def getSurfaceMesh(voxelData, ConstPixelSpacing):
    """
    Returns vertices, normals and faces by using the Marching Cubes algorithm
    """
    verts, faces, normals, values = measure.marching_cubes_lewiner(
        voxelData, 0, ConstPixelSpacing)
    return verts, normals, faces


def getOutwardNormals(normals, surfels):
    """
    Takes input of all surfels and their normals, and returns only the
    outward normals and their corresponding surfels, from the set. This
    algorithm only works for surfels from a reasonably closed point cloud.
    """
    mid = np.average(surfels, 0)
    diff_coord = surfels - mid
    outward_normals = normals[np.sum(diff_coord * normals, 1) > 0]
    outer_surfels = surfels[np.sum(diff_coord * normals, 1) > 0]
    return outward_normals, outer_surfels
