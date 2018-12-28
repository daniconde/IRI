from tkinter import *
from tkinter import filedialog
import cv2
import imutils
import numpy as np
from PIL import Image



def cropImage(file):
	# Read image
	# im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
	image = cv2.imread(file)
	
	grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Select ROI(press ENTER to obtain cropped image)
	r = cv2.selectROI("image", grayImage)
	
	 
	# Crop image
	imCrop = grayImage[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

	# Display cropped image(press ESC to close image windows)
	cv2.imshow("Image", imCrop)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

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
	for angle in np.arange(0, 360, 15):
		rotated = imutils.rotate_bound(imageROI, angle)
		cv2.imshow("Rotated (Correct)", rotated)
		cv2.waitKey(0)

def openFile():
	file = filedialog.askopenfilename(title="Abrir", initialdir="c:", filetypes=(("Ficheros PNG", "*.png"), ("Ficheros JPG", "*.jpg"), ("Todos los ficheros", "*.*")))

	print(file)

	return file

def buttonPredictPressed():
	file = openFile()
	cropImage(file)

def initializeRoot():
	# root.attributes('-fullscreen', True)
	root.state('zoomed')

	# Lo del config se puede poner directament en el constructor
	root.config(bg="blue")

def initializeFrameMainMenu():
	# filename = PhotoImage(file = "images\\background.png")
	# background_label = Label(frameMainMenu, image=filename)
	# background_label.pack(fill=BOTH, expand=1)
	buttonPredict = Button(frameMainMenu, width=200, height=200, image=icon, command=buttonPredictPressed).place(relx=.5, rely=.5, anchor=CENTER)
	
def showMainMenu():
	frameMainMenu.pack(fill=BOTH, expand=1)

def hideFrameMainMenu():
	# Esconder pack_forget() o grid_forget()
	frameMainMenu.pack_forget()

def main():
	initializeRoot()
	initializeFrameMainMenu()
	showMainMenu()
	root.mainloop()

if __name__ == '__main__':
	root = Tk()
	frameMainMenu = Frame(root)
	icon = PhotoImage(file = "images\\icon_implant.png")
	main()
