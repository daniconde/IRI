from classifier.train_model_functional_ordered import *
from classifier.package_data import *
from tkinter import *
from tkinter import filedialog
import cv2
import imutils
import numpy as np
from PIL import Image
import threading
import webbrowser



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

# def buttonMakePredictionPressed():
# 	file = openFile()
# 	result, imageRotated = rotateImage(file)
# 	if result:
# 		imageCropped = cropImage(imageRotated)
# 		imageResized = cv2.resize(imageCropped, (90, 160))
# 		cv2.imshow("Resized", imageResized)
# 		print(type(imageResized)) 

def openFile():
	file = filedialog.askopenfilename(title="Abrir", initialdir="c:", filetypes=(("Ficheros PNG", "*.png"), ("Ficheros JPG", "*.jpg"), ("Todos los ficheros", "*.*")))

	print(file)

	return file



def initializeRoot():
	# root.attributes('-fullscreen', True)
	root.state('zoomed')

	# Lo del config se puede poner directament en el constructor
	root.config(bg="blue")



def buttonMakePredictionPressed():
	hideFrameMainMenu()
	showFrameImageSelected()

def initializeFrameMainMenu():
	lblBackground = Label(frameMainMenu, image=backgroundImage)
	lblBackground.pack(fill=BOTH, expand=True)
	lblTitle = Label(lblBackground, image=titleLogo, bg="#20A099").place(relx=0.5, rely=0.2, anchor=CENTER)
	btnMakePrediction = Button(lblBackground, width=200, height=200, image=icon, command=buttonMakePredictionPressed).place(relx=.5, rely=.5, anchor=CENTER)
	
def showFrameMainMenu():
	frameMainMenu.pack(fill=BOTH, expand=True)

def hideFrameMainMenu():
	# Esconder pack_forget() o grid_forget()
	frameMainMenu.pack_forget()



def buttonOpenFilePressed(routeText, lblImageSelected, lblBottom2):
	file = openFile()
	routeText.set(file)
	photo = PhotoImage(file=file)
	lblImageSelected = Label(lblBottom2, image=photo).pack()
	lblImageSelected.image = photo
	showFrameImageSelected()

def buttonPredictPressed(routeText):
	file = routeText.get()
	image = convertOneImage(file)
	# image = cv2.imread(file)
	# grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# cv2.imshow("Show", grayImage)
	# cv2.waitKey(0)
	# grayImage = grayImage.flatten()
	pred = makePrediction(image)
	
	print(pred)
	print(type(pred))
	
	initializeFramePredictionResult(pred)
	hideFrameImageSelected()
	showFramePredictionResult()

def initializeFrameImageSelected():
	lblBackground = Label(frameImageSelected, image=backgroundImage)
	lblBackground.pack(fill=BOTH, expand=True)
	
	routeText = StringVar()
	routeText.set("...")

	lblTitle = Label(lblBackground, image=titleSeleccionarRadiografia, bg="#20A099")
	lblTitle.image = titleSeleccionarRadiografia
	lblTitle.pack(side=TOP)

	lblTop1 = Label(lblBackground, bg="#20A099")
	lblTop1.pack(side=TOP, pady=50)

	lblBottom1 = Label(lblBackground, bg="#20A099")
	lblBottom1.pack(side=BOTTOM, pady=100)

	lblTop2 = Label(lblTop1, bg="#20A099")
	lblTop2.pack(side=TOP)

	lblBottom2 = Label(lblTop1, bg="#20A099")
	lblBottom2.pack(side=BOTTOM)

	# lblTitleFileSelected = Label(lblTop2, text="Ruta archivo:").pack(side=TOP)
	lblRouteFileSelected = Label(lblTop2, textvariable=routeText, background="white").pack(side=LEFT, pady=20)
	lblImageSelected = Label(lblBottom2)
	btnOpenFile = Button(lblTop2, text="Abrir archivo", command=lambda:buttonOpenFilePressed(routeText, lblImageSelected, lblBottom2)).pack(side=LEFT, padx=10, pady=20)
	btnPredict = Button(lblBottom1, text="Realizar predicción", font=20, command=lambda:buttonPredictPressed(routeText)).pack()
	
def showFrameImageSelected():
	frameImageSelected.pack(fill=BOTH, expand=True)

def hideFrameImageSelected():
	# Esconder pack_forget() o grid_forget()
	frameImageSelected.pack_forget()



def callback(event, model):
	if model == "Astra": 
		webbrowser.open_new(r"https://www.dentsplysirona.com/es-ib/productos/implantes/soluciones-con-implantes/astra-tech-implant-system-ev.html")
	elif model == "NobelActive": 
		webbrowser.open_new(r"https://store.nobelbiocare.com/es/es/implantes/nobelactive")
	elif model == "NobelParallel": 
		webbrowser.open_new(r"https://store.nobelbiocare.com/es/es/implantes/nobelparallel-cc")
	elif model == "NobelReplaceSelect": 
		webbrowser.open_new(r"https://store.nobelbiocare.com/es/es/implantes/nobelreplaceselect-straight/replace-select-tc")
	elif model == "NobelReplaceTapered": 
		webbrowser.open_new(r"https://store.nobelbiocare.com/es/es/implantes/nobelreplaceselect-tapered/replace-select-tapered")
	elif model == "NobelSpeedyReplaceTrichannel": 
		webbrowser.open_new(r"https://store.nobelbiocare.com/es/es/implantes/nobelspeedy")

def initializeFramePredictionResult(pred):
	lblBackground = Label(framePredictionResult, image=backgroundImage)
	lblBackground.pack(fill=BOTH, expand=True)

	lblTitle = Label(lblBackground, image=titleResultadosPrediccion, bg="#20A099")
	lblTitle.place(relx=0.5, rely=0.1, anchor=CENTER)

	lblCenterWidgets = Label(lblBackground, image=backgroundImage)
	lblCenterWidgets.place(relx=0.5, rely=0.6, anchor=CENTER)

	lblTitleName = Label(lblCenterWidgets, image=titleModelo, bg="#20A099").grid(row=0, column=0)
	lblTitleProb = Label(lblCenterWidgets, image=titleProbabilidad, bg="#20A099").grid(row=0, column=1)
	lblTitleLink = Label(lblCenterWidgets, image=titleLink, bg="#20A099").grid(row=0, column=2)

	lblAstraName = Label(lblCenterWidgets, text="Astra", bg="#20A099", font=("Arial", 20)).grid(row=1, column=0, pady=20)
	lblAstraProb = Label(lblCenterWidgets, text=pred.item(0), bg="#20A099", font=("Arial", 20)).grid(row=1, column=1, padx=20, pady=20)
	lblAstraLink = Label(lblCenterWidgets, text="Astra - OsseoSpeed EV", bg="#20A099", font=("Arial", 20), cursor="hand2")
	lblAstraLink.grid(row=1, column=2, padx=20, pady=20)
	lblAstraLink.bind("<Button-1>", lambda event:callback(event, "Astra"))

	lblNobelActiveName = Label(lblCenterWidgets, text="Nobel Active", bg="#20A099", font=("Arial", 20)).grid(row=2, column=0, pady=20)
	lblNobelActiveProb = Label(lblCenterWidgets, text=pred.item(1), bg="#20A099", font=("Arial", 20)).grid(row=2, column=1, padx=20, pady=20)
	lblNobelActiveLink = Label(lblCenterWidgets, text="Nobel - Active", bg="#20A099", font=("Arial", 20), cursor="hand2")
	lblNobelActiveLink.grid(row=2, column=2, padx=20, pady=20)
	lblNobelActiveLink.bind("<Button-1>", lambda event:callback(event, "NobelActive"))

	lblNobelParallelName = Label(lblCenterWidgets, text="Nobel Parallel", bg="#20A099", font=("Arial", 20)).grid(row=3, column=0, pady=20)
	lblNobelParallelProb = Label(lblCenterWidgets, text=pred.item(2), bg="#20A099", font=("Arial", 20)).grid(row=3, column=1, padx=20, pady=20)
	lblNobelParallelLink = Label(lblCenterWidgets, text="Nobel - Parallel CC", bg="#20A099", font=("Arial", 20), cursor="hand2")
	lblNobelParallelLink.grid(row=3, column=2, padx=20, pady=20)
	lblNobelParallelLink.bind("<Button-1>", lambda event:callback(event, "NobelParallel"))

	lblNobelReplaceSelectName = Label(lblCenterWidgets, text="Nobel Replace Select", bg="#20A099", font=("Arial", 20)).grid(row=4, column=0, pady=20)
	lblNobelReplaceSelectProb = Label(lblCenterWidgets, text=pred.item(3), bg="#20A099", font=("Arial", 20)).grid(row=4, column=1, padx=20, pady=20)
	lblNobelReplaceSelectLink = Label(lblCenterWidgets, text="Nobel - Replace Select TC", bg="#20A099", font=("Arial", 20), cursor="hand2")
	lblNobelReplaceSelectLink.grid(row=4, column=2, padx=20, pady=20)
	lblNobelReplaceSelectLink.bind("<Button-1>", lambda event:callback(event, "NobelReplaceSelect"))

	lblNobelReplaceTaperedName = Label(lblCenterWidgets, text="Nobel Replace Tapered", bg="#20A099", font=("Arial", 20)).grid(row=5, column=0, pady=20)
	lblNobelReplaceTaperedProb = Label(lblCenterWidgets, text=pred.item(4), bg="#20A099", font=("Arial", 20)).grid(row=5, column=1, padx=20, pady=20)
	lblNobelReplaceTaperedLink = Label(lblCenterWidgets, text="Nobel - Replace Tapered", bg="#20A099", font=("Arial", 20), cursor="hand2")
	lblNobelReplaceTaperedLink.grid(row=5, column=2, padx=20, pady=20)
	lblNobelReplaceTaperedLink.bind("<Button-1>", lambda event:callback(event, "NobelReplaceTapered"))

	lblNobelSpeedyReplaceTrichannelName = Label(lblCenterWidgets, text="Nobel Speedy Replace Trichannel", bg="#20A099", font=("Arial", 20)).grid(row=6, column=0, pady=20)
	lblNobelSpeedyReplaceTrichannelProb = Label(lblCenterWidgets, text=pred.item(5), bg="#20A099", font=("Arial", 20)).grid(row=6, column=1, padx=20, pady=20)
	lblNobelSpeedyReplaceTrichannelLink = Label(lblCenterWidgets, text="Nobel - Speedy Replace TriChannel", bg="#20A099", font=("Arial", 20), cursor="hand2")
	lblNobelSpeedyReplaceTrichannelLink.grid(row=6, column=2, padx=20, pady=20)
	lblNobelSpeedyReplaceTrichannelLink.bind("<Button-1>", lambda event:callback(event, "NobelSpeedyReplaceTrichannel"))

def showFramePredictionResult():
	framePredictionResult.pack(fill=BOTH, expand=True)

def hideFramePredictionResult():
	# Esconder pack_forget() o grid_forget()
	framePredictionResult.pack_forget()



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
	framePredictionResult = Frame(root)
	backgroundImage = PhotoImage(file = "presentation\\images\\background.png")
	titleLogo = PhotoImage(file = "presentation\\images\\logo.png")
	titleSeleccionarRadiografia = PhotoImage(file = "presentation\\images\\seleccionar_radiografia.png")
	titleResultadosPrediccion = PhotoImage(file = "presentation\\images\\resultados_prediccion.png")
	titleModelo = PhotoImage(file = "presentation\\images\\modelo.png")
	titleProbabilidad = PhotoImage(file = "presentation\\images\\probabilidad.png")
	titleLink = PhotoImage(file = "presentation\\images\\link.png")
	icon = PhotoImage(file = "presentation\\images\\icon_implant.png")
	main()
