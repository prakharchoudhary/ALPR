"""
Segmentation to find all those parts of the image that could
have a plate.
"""

from copy import deepcopy, copy
import cv2
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pytesseract
import os
import random

class Segmentation:

	def __init__(self, img):
		'''
		img is an image matrix built using cv2.imread() 
		'''
		self.img = img
		self.plate_loc = deepcopy(img)
		self.rects = []

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
		# cv2.imshow("After sobel", self.img_sobel)
		# cv2.waitKey(0)

	def _laplacian(self):
		'''
		Look for edges(found abundantly in number plates)
		using Laplacian filter.
		'''
		self.img_laplacian = cv2.Laplacian(self.gray_img, cv2.CV_8U)
		ret, self.img_thresh = cv2.threshold(self.img_laplacian, 0, 255, 
			cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		# cv2.imshow("After filter", self.img_laplacian)
		# cv2.waitKey(0)		

	def _morph(self):
		'''
		Create a rectangular structural element and apply 
		morphological operation.
		'''
		element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
		self.img_thresh = cv2.morphologyEx(self.img_thresh, cv2.MORPH_CLOSE,
			element)
		# cv2.imshow("After morphing", self.img_thresh)
		# cv2.waitKey(0)

	def plateSearch(self):
		'''
		Method to find plate and read characters
		'''
		crop = self.cropPlate();
		# if self.plate_image is not None:
		# 	self.readPlateNumber(characters_array);
		# print(self.plate_number)
		if crop:
			self.showResults();
			return True;
		return False

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
		# self._sobel()
		self._laplacian()
		# self._morph()
		self.find_contours()
		self.allRects = []
		for contour in self.contours:
			# self.allRects.append(cv2.minAreaRect(contour))
			self.allRects.append(cv2.boundingRect(contour))

	def verifySizes(self, rect):
		'''
		basic validations about the regions detected based on its 
		area and aspect ratio.
		'''
		ERROR = 0.4
		ASPECT = 4.772
		min_area = int(75*ASPECT*75)
		max_area = int(125*ASPECT*125)
		rmin = ASPECT - ASPECT*ERROR
		rmax = ASPECT + ASPECT*ERROR

		# (x, y), (width, height), rect_angle = rect
		x, y, width, height = rect

		# print(rect)
		area = height * width
		try:
			r = float(width) / height
			if r<1:
				r = float(height) / width
		except ZeroDivisionError:
			# print("Contour discarded")
			return False

		if (area<min_area or area > max_area) or (r < rmin or r> max_area):
			return False

		return True

	def _getCenter(self, rect):
		'''
		obtain center of the rect
		'''
		# (x, y), (width, height), rect_angle = rect
		x, y, width, height = rect
		return (int((x+width)/2), int((y+height)/2))

	def get_crop_mask(self):
		'''
		Using floodfill Algorithm to retreive the contour box more clearly.
		'''
		self.get_rects()
		result = deepcopy(self.img)
		cv2.drawContours(result, self.contours,
			-1,
			(0,0,255),
			3)
		# cv2.imshow('contour', result)
		# cv2.waitKey(0)

		# get all valid rects
		for rect in self.allRects:
			if self.verifySizes(rect):
				self.rects.append(rect)

		img_boxed = deepcopy(self.img)
		for rect in self.rects:
			# (x, y), (w, h), rect_angle = rect
			x, y, w, h = rect
			cv2.rectangle(img_boxed, (int(x),int(y)), 
				(int(x+w), int(y+h)), (0,0,255), 1)
			cv2.imshow("Boxed image", img_boxed)
			cv2.waitKey(0)

		for idx, rect in enumerate(self.rects):
			# (x, y), (width, height), rect_angle = rect
			x, y, width, height = rect
			rect_center = self._getCenter(rect)
			cv2.circle(result, rect_center, 3, (0, 255, 0), -1)
			if width < height:
				minSize = width
			else:
				minSize = height

			minSize = minSize*0.5
			#Initialize floodfill parameters and variables
			mask = np.zeros((self.img.shape[0] + 2, 
				self.img.shape[1] + 2, 1), dtype="uint8")
			loDiff = 30
			upDiff = 30
			connectivity = 4
			newMaskVal = 255
			NumSeeds = 10
			flags = connectivity + (newMaskVal << 8) + cv2.FLOODFILL_FIXED_RANGE + \
				cv2.FLOODFILL_MASK_ONLY
			for j in range(NumSeeds):
				seed = (rect_center[0] + random.randint(0, 32767)%int(minSize) - int(minSize/2),
					rect_center[1] + random.randint(0, 32767)%int(minSize) - int(minSize/2))
				cv2.circle(self.img, seed, 1, (0,255,255), -1)
				area = cv2.floodFill(self.img, mask, seed, (255,0,0),
					(loDiff,)*3, (upDiff,)*3, flags)

	def getPoints(self, point):
		'''
		transform the rect tuple: ((h,w), (x,y), angle) into 
		[x,y,w,h]
		'''
		print(point)
		x = map(int, [point[1][0],
				point[1][1],
				point[0][1],
				point[0][0]
			])
		return x

	def cropPlate(self):
		'''
		If a license plate is found
		'''
		self.roi = []
		self.get_rects()
		self.get_crop_mask()
		for rect in self.rects:
			if(self.verifySizes(rect)):
				self.roi.append(rect)
		if len(self.roi)>1:
			# [x,y,w,h] = self.getPoints(self.roi[0])
			[x,y,w,h] = self.roi[0]
			print(x,y,w,h)
			print((x,y), (x+w, y+h))
		else:
			return False
		cv2.rectangle(self.plate_loc, (x,y), (x+w, y+h), (0,0,255), 3)
		self.plate_image = self.img[y:y+h,x:x+w]
		# cv2.imshow("Thresh image" ,self.img_thresh)
		# cv2.waitKey(0)
		cv2.imwrite("output.png", self.plate_image)
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

		cv2.destroyAllWindows()
		plt.tight_layout()
		plt.show()
		return True

