import os
import pytesseract
from PIL import Image
import cv2
from plate import Segmentation

if __name__ == "__main__":
	
	images = os.listdir('testData/')
	print(images)
	for image in images:
		img_path = "./testData/{}".format(image)
		if os.path.exists(os.path.join(os.getcwd(), "./output.png")):
			os.system("rm ./output.png")
		image = cv2.imread(img_path)
		seg = Segmentation(image)
		img = seg.plateSearch()
		if img:
			# text = pytesseract.image_to_string(Image.fromarray(img))
			text = pytesseract.image_to_string(Image.open("./output.png"))
			print(text)
		else:
			print("No number plate found")