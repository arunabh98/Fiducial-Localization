from math import *
import numpy as np
from mayavi import mlab

def genFiducialPC(dist = 0.3):
	# Dimensions in mm
	R1 = 2
	R2 = 5.5
	H  = 2.83

	allPoints = []

	# CURVED SURFACE
	dTheta1 = dist/R1
	dTheta2 = dist/R2
	for h in np.linspace(0,H,ceil(H/dist)+1):
		allPoints.extend([[R1*cos(theta1), R1*sin(theta1), h] for theta1 in np.arange(0,2*pi,dTheta1)])
		allPoints.extend([[R2*cos(theta2), R2*sin(theta2), h] for theta2 in np.arange(0,2*pi,dTheta2)])

	# FLAT SURFACE
	allPoints.append((0,0,0))
	for r in np.linspace(dist,R2,ceil(R2/dist)+1):
		dTheta = dist/r
		if r < R1:
			allPoints.extend([[r*cos(theta), r*sin(theta), 0] for theta in np.arange(0,2*pi,dTheta)])
		if r > R1:
			allPoints.extend([[r*cos(theta), r*sin(theta), H] for theta in np.arange(0,2*pi,dTheta)])
	allPoints = np.asarray(allPoints)
	mlab.points3d(allPoints[:,0],allPoints[:,1],allPoints[:,2])
	return allPoints

if __name__ == '__main__' :
	allPoints = genFiducialPC()
	print len(allPoints)

	from mayavi import mlab

	X = allPoints[:,0]
	Y = allPoints[:,1]
	Z = allPoints[:,2]

	mlab.points3d(X, Y, Z, mode='point', scale_factor=.25)
	mlab.show()
