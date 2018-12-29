from classifier.train_model_functional_ordered import *
from tkinter import *
from tkinter import filedialog
import cv2
import imutils
import numpy as np
from PIL import Image



def cropImage(image):
	# Read image
	# im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
	# image = cv2.imread(file)
	
	grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Select ROI(press ENTER to obtain cropped image)
	r = cv2.selectROI("image", grayImage)
	
	 
	# Crop image
	imCrop = grayImage[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

	# Display cropped image(press ESC to close image windows)
	cv2.imshow("Image", imCrop)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return imCrop

def rotateImage(file):
	# Read image
	image = cv2.imread(file)
	
	grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	edged = cv2.Canny(grayImage, 20, 100)

	# find contours in the edge map
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# ensure at least one contour was found
	if len(cnts) > 0:
		# grab the largest contour, then draw a mask for the pill
		c = max(cnts, key=cv2.contourArea)
		mask = np.zeros(grayImage.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)
	 
		# compute its bounding box of pill, then extract the ROI,
		# and apply the mask
		(x, y, w, h) = cv2.boundingRect(c)
		imageROI = image[y:y + h, x:x + w]
		maskROI = mask[y:y + h, x:x + w]
		imageROI = cv2.bitwise_and(imageROI, imageROI,
			mask=maskROI)


	# loop over the rotation angles again, this time ensure the
	# entire pill is still within the ROI after rotation
	# for angle in np.arange(0, 360, 15):
	# 	rotated = imutils.rotate_bound(imageROI, angle)
	# 	cv2.imshow("Rotated (Correct)", rotated)
	# 	cv2.waitKey(0)

	angle = 0
	f = True
	# k = 27 --> ESC
	while (f):
		rotated = imutils.rotate_bound(imageROI, angle)
		cv2.namedWindow("Rotated", cv2.WINDOW_AUTOSIZE)
		# cv2.resizeWindow("Rotated", 500, 500)
		# cv2.moveWindow("Rotated", 100, 100)
		cv2.imshow("Rotated", rotated)
		k = cv2.waitKey(0)
		if (k == ord('d')):
			angle += 15
		if (k == ord('a')):
			angle -= 15
		if k == 27:
			cv2.destroyAllWindows()
			f = False
		if k == ord('o'):
			cv2.destroyAllWindows()
			f = False
			return True, rotated
	return False, None

	# if k == 27:
	# 	cv2.destroyAllWindows()
	# elif k == ord('\n'):
	# 	cv2.destroyAllWindows()

def openFile():
	file = filedialog.askopenfilename(title="Abrir", initialdir="c:", filetypes=(("Ficheros PNG", "*.png"), ("Ficheros JPG", "*.jpg"), ("Todos los ficheros", "*.*")))

	print(file)

	return file

# def buttonMakePredictionPressed():
# 	file = openFile()
# 	result, imageRotated = rotateImage(file)
# 	if result:
# 		imageCropped = cropImage(imageRotated)
# 		imageResized = cv2.resize(imageCropped, (90, 160))
# 		cv2.imshow("Resized", imageResized)
# 		print(type(imageResized)) 

def buttonMakePredictionPressed():
	hideFrameMainMenu()
	showFrameImageSelected()

def initializeRoot():
	# root.attributes('-fullscreen', True)
	root.state('zoomed')

	# Lo del config se puede poner directament en el constructor
	root.config(bg="blue")

def initializeFrameMainMenu():
	# filename = PhotoImage(file = "images\\background.png")
	# background_label = Label(frameMainMenu, image=filename)
	# background_label.pack(fill=BOTH, expand=1)
	btnMakePrediction = Button(frameMainMenu, width=200, height=200, image=icon, command=buttonMakePredictionPressed).place(relx=.5, rely=.5, anchor=CENTER)
	
def showFrameMainMenu():
	frameMainMenu.pack(fill=BOTH, expand=1)

def hideFrameMainMenu():
	# Esconder pack_forget() o grid_forget()
	frameMainMenu.pack_forget()

def buttonOpenFilePressed(routeText, lblImageSelected):
	file = openFile()
	routeText.set(file)
	photo = PhotoImage(file=file)
	lblImageSelected = Label(frameImageSelected, image=photo).grid(row=1, column=0, columnspan=1)
	lblImageSelected.photo = photo

def buttonPredictPressed(routeText):
	file = routeText.get()
	image = cv2.imread(file)
	# grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.imshow("Show", image)
	cv2.waitKey(0)
	pred = makePrediction(image)


def initializeFrameImageSelected():
	# filename = PhotoImage(file = "images\\background.png")
	# background_label = Label(frameMainMenu, image=filename)
	# background_label.pack(fill=BOTH, expand=1)
	routeText = StringVar()
	lblTitleFileSelected = Label(frameImageSelected, text="Ruta archivo:").grid(row=0, column=0)
	entRouteFileSelected = Entry(frameImageSelected, textvariable=routeText, background="white").grid(row=0, column=1)
	lblImageSelected = Label(frameImageSelected).grid(row=1, column=0, columnspan=1)
	btnOpenFile = Button(frameImageSelected, text="Abrir archivo", command=lambda:buttonOpenFilePressed(routeText, lblImageSelected)).grid(row=0, column=3)
	btnPredict = Button(frameImageSelected, text="Realizar predicción", command=lambda:buttonPredictPressed(routeText)).grid(row=2, column=0)

	
def showFrameImageSelected():
	frameImageSelected.pack(fill=BOTH, expand=1)

def hideFrameImageSelected():
	# Esconder pack_forget() o grid_forget()
	frameImageSelected.pack_forget()

def main():
	initializeRoot()
	initializeFrameMainMenu()
	initializeFrameImageSelected()
	showFrameMainMenu()
	root.mainloop()

if __name__ == '__main__':
	root = Tk()
	frameMainMenu = Frame(root)
	frameImageSelected = Frame(root)
	icon = PhotoImage(file = "presentation\\images\\icon_implant.png")
	main()
