"""
Segmentation to find all those parts of the image that could
have a plate.
"""

import cv2
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy, copy
import random

class Segmentation:

	def __init__(self, img):
		'''
		img is an image matrix built using cv2.imread() 
		'''
		self.img = img
		self.plate_loc = deepcopy(img)

	def _to_grayscale(self):
		'''
		Convert the color image to grayscale and apply
		Gaussian blur(5 x 5) to remove noise.
		'''
		self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		self.gray_img = cv2.GaussianBlur(self.gray_img, (5, 5), 0)

	def _sobel(self):
		'''
		Look for vertical edges(found abundantly in number plates)
		'''
		self.img_sobel = cv2.Sobel(self.gray_img, cv2.CV_8U, 1, 0, ksize=3)
		ret, self.img_thresh = cv2.threshold(self.img_sobel, 0, 255, 
			cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	def _morph(self):
		'''
		Create a rectangular structural element and apply 
		morphological operation.
		'''
		element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
		self.img_thresh = cv2.morphologyEx(self.img_thresh, cv2.MORPH_CLOSE,
			element)

	def plateSearch(self):
		'''
		Method to find plate and read characters
		'''
		self.cropPlate();
		# if self.plate_image is not None:
		# 	self.readPlateNumber(characters_array);
		# print(self.plate_number)
		self.showResults();
		return True;

	def find_contours(self):
		'''
		Find all relevant contours
		'''
		_, self.contours, self.hierarchy = cv2.findContours(
			self.img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	def get_rects(self):
		'''
		Extract a bounding a rectangle of minimal area for each 
		relevant contours.
		'''
		self._to_grayscale()
		self._sobel()
		self._morph()
		self.find_contours()
		self.rects = []
		for contour in self.contours:
			self.rects.append(cv2.minAreaRect(contour))

	def verifySizes(self, rect):
		'''
		basic validations about the regions detected based on its 
		area and aspect ratio.
		'''
		ERROR = 0.4
		ASPECT = 4.116 # since 500x120 aspect is found in indian number plates
		min_area = int(15*ASPECT*15)
		max_area = int(125*ASPECT*125)
		rmin = ASPECT - ASPECT*ERROR
		rmax = ASPECT + ASPECT*ERROR

		# print(rect)
		area = rect[0][0] * rect[0][1]
		r = float(rect[0][0]) / rect[0][1]
		if r<1:
			r = 1.0/r

		if (area<min_area or area > max_area) or (r < rmin or r> max_area):
			return False

		else:
			return True

	# def get_crop_mask(self):
	# 	'''
	# 	Using floodfill Algorithm to retreive the contour box more clearly.
	# 	'''
	# 	self.get_rects()
	# 	for idx, rect in enumerate(self.rects):
	# 		cv2.circle(self.image_thresh, rect.center, 3, (0, 255, 0), -1)
	# 		if rect.size.width < rect.size.width:
	# 			minSize = rect.size.width
	# 		else:
	# 			minSize = rect.size.height

	# 		minSize = minSize*0.5
	# 		#Initialize floodfill parameters and variables
	# 		mask = np.zeros((self.img_thresh.rows + 2, 
	# 			self.img_thresh.cols + 2, 1), dtype="uint8")
	# 		loDiff = 30
	# 		upDiff = 30
	# 		connectivity = 4
	# 		newMaskVal = 255
	# 		NumSeeds = 10
	# 		flags = connectivity + (newMaskVal << 8) + cv2.FLOODFILL_FIXED_RANGE + \
	# 			cv2.CV_FLOODFILL_MASK_ONLY
	# 		for j in range(NumSeeds):
	# 			seed = [rect.center.x + random.randint()%int(minSize) - (minSize/2),
	# 				rect.center.y + random.randint()%int(minSize) - (minSize/2)]
	# 			cv2.circle(self.image_thresh, seed, 1, (0,255,255), -1)
	# 			area = cv2.floodFill(self.img, mask, seed, (255,0,0),
	# 				(loDiff,)*3, (upDiff,)*3, flags)
	# 	return mask

	def getPoints(self, point):
		'''
		transform the rect tuple: ((h,w), (x,y), angle) into 
		[x,y,w,h]
		'''
		return map(int, [point[1][0],
				point[1][1],
				point[0][1],
				point[0][0]
			])

	def cropPlate(self):
		'''
		If a license plate is found
		'''
		self.roi = []
		self.get_rects()
		# mask = self.get_crop_mask()
		for rect in self.rects:
			if(self.verifySizes(rect)):
				self.roi.append(rect)
		if len(self.roi)>1:
			[x,y,w,h] = self.getPoints(self.roi[0])
			cv2.rectangle(self.plate_loc, (x,y), (x+w, y+h), (0,0,255), 1)
			self.plate_image = self.img[y:y+h,x:x+w]
		return True

	def plot(self, figure, subplot, image, title):
		figure.subplot(subplot);
		figure.imshow(image);
		figure.xlabel(title);
		figure.xticks([]);
		figure.yticks([]);
		return True;

	def showResults(self):
		# plt.figure(self.plate_number)

		self.plot(plt, 321, self.img, "Original image");
		self.plot(plt, 322, self.gray_img, "Threshold image");
		self.plot(plt, 323, self.plate_loc, "Plate located");

		if self.plate_image is not None:
			self.plot(plt, 324, self.plate_image, "License plate");

		plt.tight_layout()
		plt.show()
		return True

if __name__ == "__main__":
	img_path = "./images/test.jpg"
	image = cv2.imread(img_path)
	seg = Segmentation(image)
	seg.plateSearch()