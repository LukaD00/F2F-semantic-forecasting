import numpy as np

def mIoU(nparray1, nparray2, classes, void_class = 255):
	"""
	Args:
		nparray1 (np.ndarray) - first array
		nparray2 (np.ndarray) - second array
		classes ([int]) - class labels to calculate mIoU for

	Returns:
		mean Intersection over Union between two arrays for all given classes
	"""

	# initialize Intersection-Union dict
	IUs = {}
	for c in classes:
		IUs[c] = [0,0] # c -> (c_intersection, c_union)

	# calculate intersections and unions for each class at once
	for pixel1, pixel2 in np.nditer([nparray1, nparray2]):
		c1 = pixel1.item()
		c2 = pixel2.item()
		if c1 == void_class or c2 == void_class:
			continue
		if c1 == c2: # intersection
			if c1 in IUs:
				IUs[c1][0] += 1
				IUs[c1][1] += 1
		else: # union
			if c1 in IUs: IUs[c1][1] += 1
			if c2 in IUs: IUs[c2][1] += 1
	
	# calculate intersection/union for each class
	IoUs = []
	for c in IUs:
		if IUs[c][1] == 0: # no such pixels found, would be 0/0 but we declare 1
			IoUs.append(1)
		else:
			IoUs.append(IUs[c][0] / IUs[c][1])

	return np.nanmean(IoUs)
