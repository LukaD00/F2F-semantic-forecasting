import numpy as np

def IoU(nparray1, nparray2, class_number):
	"""
	Args:
		nparray1 (np.array) - first array
		nparray2 (np.array) - second array
		class_number (int) - class label to calculate IoU for

	Returns:
		Intersection over Union between two arrays for given class
	"""

	intersection = 0
	union = 0
	for pixel1, pixel2 in np.nditer([nparray1, nparray2]):
		if pixel1 == class_number and pixel2 == class_number:
			intersection += 1
		if pixel1 == class_number or pixel2 == class_number:
			union += 1
	return 1.0*intersection/union

def mIoU(nparray1, nparray2, classes):
	"""
	Args:
		nparray1 (np.array) - first array
		nparray2 (np.array) - second array
		classes ([int]) - class labels to calculate mIoU for

	Returns:
		mean Intersection over Union between two arrays for all given classes
	"""

	ious = []
	for class_number in classes:
		ious.append(IoU(nparray1, nparray2, class_number))
	ious.sort()
	return np.nanmean(ious)
