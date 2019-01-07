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
import locale
# import screeninfo



def cropImage():
	# Read image
	# im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
	image = cv2.imread('radiography.png')
	
	grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Select ROI(press ENTER to obtain cropped image)
	r = cv2.selectROI("image", grayImage)
	
	 
	# Crop image
	imCrop = grayImage[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

	# Display cropped image(press ESC to close image windows)
	# screen_id = 2
	# screen = screeninfo.get_monitors()[screen_id]
	# cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
	# cv2.moveWindow("Image", screen.x - 1, screen.y - 1)
	# cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	cv2.imshow("Image", imCrop)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return imCrop

def rotateImage():
	# Read image
	image = cv2.imread('radiography.png')
	
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
	destroyFrameMainMenu()
	global frameImageSelected
	frameImageSelected = Frame(root)
	initializeFrameImageSelected("...", None)
	showFrameImageSelected()

def initializeFrameMainMenu():
	lblBackground = Label(frameMainMenu, image=backgroundImage)
	lblBackground.pack(fill=BOTH, expand=True)
	titleLogo = PhotoImage(file = "presentation\\images\\logo.png")
	lblTitle = Label(lblBackground, image=titleLogo, bg="#20A099")
	lblTitle.image = titleLogo
	lblTitle.place(relx=0.5, rely=0.2, anchor=CENTER)
	iconSearchImplant = PhotoImage(file = "presentation\\images\\icon_implant.png")
	btnMakePrediction = Button(lblBackground, width=200, height=200, image=iconSearchImplant, command=buttonMakePredictionPressed)
	btnMakePrediction.image = iconSearchImplant
	btnMakePrediction.place(relx=.5, rely=.5, anchor=CENTER)
	
def showFrameMainMenu():
	frameMainMenu.pack(fill=BOTH, expand=True)

def hideFrameMainMenu():
	# Esconder pack_forget() o grid_forget()
	frameMainMenu.pack_forget()

def destroyFrameMainMenu():
	frameMainMenu.destroy()



def buttonBackFrameImageSelectedPressed():
	destroyFrameImageSelected()
	global frameMainMenu
	frameMainMenu = Frame(root)
	initializeFrameMainMenu()
	showFrameMainMenu()

def buttonOpenFilePressed():
	file = openFile()
	
	# height -> image.shape[0], width -> image.shape[1], channels -> image.shape[2]
	image = cv2.imread(file)
	# height -> image.shape[0], width -> image.shape[1], channels -> no tiene
	grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	imageResized = cv2.resize(grayImage, (int(grayImage.shape[1]*0.6), int(grayImage.shape[0]*0.6)))

	cv2.imwrite('radiography.png', imageResized)

	photo = PhotoImage(file='radiography.png')
	destroyFrameImageSelected()
	global frameImageSelected
	frameImageSelected = Frame(root)
	initializeFrameImageSelected(file, photo)
	showFrameImageSelected()

def buttonRotatePressed(file):
	result, image = rotateImage()
	if result:
		cv2.imwrite('radiography.png', image)
		photo = PhotoImage(file='radiography.png')
		destroyFrameImageSelected()
		global frameImageSelected
		frameImageSelected = Frame(root)
		initializeFrameImageSelected(file, photo)
		showFrameImageSelected()

def buttonCropPressed(file):
	image = cropImage()
	cv2.imwrite('radiography.png', image)
	photo = PhotoImage(file='radiography.png')
	destroyFrameImageSelected()
	global frameImageSelected
	frameImageSelected = Frame(root)
	initializeFrameImageSelected(file, photo)
	showFrameImageSelected()

def buttonScalePressed(file):
	image = cv2.imread('radiography.png')
	grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	imageResized = cv2.resize(grayImage, (90, 160))
	cv2.imwrite('radiography.png', imageResized)
	photo = PhotoImage(file='radiography.png')
	destroyFrameImageSelected()
	global frameImageSelected
	frameImageSelected = Frame(root)
	initializeFrameImageSelected(file, photo)
	showFrameImageSelected()


def buttonPredictPressed():
	image = convertOneImage('radiography.png')
	# image = cv2.imread(file)
	# grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# cv2.imshow("Show", grayImage)
	# cv2.waitKey(0)
	# grayImage = grayImage.flatten()
	pred = makePrediction(image)
	
	print(pred)
	print(type(pred))
	
	destroyFrameImageSelected()
	global framePredictionResult
	framePredictionResult = Frame(root)
	initializeFramePredictionResult(pred)
	showFramePredictionResult()

def initializeFrameImageSelected(text, photo):
	lblBackground = Label(frameImageSelected, image=backgroundImage)
	lblBackground.pack(fill=BOTH, expand=True)
	
	routeText = StringVar()
	routeText.set(text)

	titleSeleccionarRadiografia = PhotoImage(file = "presentation\\images\\seleccionar_radiografia.png")
	lblTitle = Label(lblBackground, image=titleSeleccionarRadiografia, bg="#20A099")
	lblTitle.image = titleSeleccionarRadiografia
	lblTitle.pack(side=TOP)

	iconBack = PhotoImage(file = "presentation\\images\\left_arrow.png")
	btnBack = Button(lblBackground, image=iconBack, bg="#20A099", border="0", width=90, height=90, command=lambda:buttonBackFrameImageSelectedPressed())
	btnBack.image = iconBack
	btnBack.pack(side=TOP, anchor=W, padx=10)

	lblTop1 = Label(lblBackground, bg="#20A099")
	lblTop1.pack(side=TOP, pady=30)

	# Lo utilizaba btnPredict
	# lblBottom1 = Label(lblBackground, bg="#20A099")
	# lblBottom1.pack(side=BOTTOM, pady=100)

	lblTop2 = Label(lblTop1, bg="#20A099")
	lblTop2.pack(side=TOP)

	lblRight2 = Label(lblTop1, bg="#20A099")
	lblRight2.pack(side=RIGHT)

	lblBottom2 = Label(lblTop1, bg="#20A099")
	lblBottom2.pack(side=BOTTOM)

	# lblTitleFileSelected = Label(lblTop2, text="Ruta archivo:").pack(side=TOP)
	lblRouteFileSelected = Label(lblTop2, textvariable=routeText, background="white").pack(side=LEFT, pady=20)
	if photo != None:
		lblImageSelected = Label(lblBottom2, image=photo)
		lblImageSelected.image = photo
		lblImageSelected.pack()
	btnOpenFile = Button(lblTop2, text="Abrir archivo", command=lambda:buttonOpenFilePressed()).pack(side=LEFT, padx=10, pady=20)
	if photo == None:
		btnPredict = Button(lblBackground, text="Realizar predicción", font=20, state=DISABLED, command=lambda:buttonPredictPressed(routeText)).pack(side=BOTTOM, pady=100)
	else:
		btnPredict = Button(lblBackground, text="Realizar predicción", font=20, command=lambda:buttonPredictPressed()).pack(side=BOTTOM, pady=100)
		btnRotate = Button(lblRight2, text="Rotar imagen", command=lambda:buttonRotatePressed(text)).pack(side=TOP)
		btnCrop = Button(lblRight2, text="Recortar imagen", command=lambda:buttonCropPressed(text)).pack(side=TOP, pady=10)
		btnScale = Button(lblRight2, text="Escalar imagen", command=lambda:buttonScalePressed(text)).pack(side=TOP, pady=10)

def showFrameImageSelected():
	frameImageSelected.pack(fill=BOTH, expand=True)

def hideFrameImageSelected():
	# Esconder pack_forget() o grid_forget()
	frameImageSelected.pack_forget()

def destroyFrameImageSelected():
	frameImageSelected.destroy()



def buttonBackFramePredictionResult():
	destroyFramePredictionResult()
	global frameImageSelected
	frameImageSelected = Frame(root)
	initializeFrameImageSelected("...", None)
	showFrameImageSelected()

def buttonMainMenuPressed():
	destroyFramePredictionResult()
	global frameMainMenu
	frameMainMenu = Frame(root)
	initializeFrameMainMenu()
	showFrameMainMenu()

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

	# titleResultadosPrediccion = PhotoImage(file = "presentation\\images\\resultados_prediccion.png")
	# lblTitle = Label(lblBackground, image=titleResultadosPrediccion, bg="#20A099")
	# lblTitle.image = titleResultadosPrediccion
	# lblTitle.place(relx=0.5, rely=0.1, anchor=CENTER)

	lblTitle = Label(lblBackground, image=titleResultadosPrediccion, bg="#20A099")
	lblTitle.pack(side=TOP)

	iconBack = PhotoImage(file = "presentation\\images\\left_arrow.png")
	btnBack = Button(lblBackground, image=iconBack, bg="#20A099", border="0", width=90, height=90, command=lambda:buttonBackFramePredictionResult())
	btnBack.image = iconBack
	btnBack.pack(side=TOP, anchor=W, padx=10)

	lblCenterWidgets = Label(lblBackground, image=backgroundImage)
	lblCenterWidgets.pack(side=TOP)

	titleModelo = PhotoImage(file = "presentation\\images\\modelo.png")
	lblTitleName = Label(lblCenterWidgets, image=titleModelo, bg="#20A099")
	lblTitle.image = titleModelo
	lblTitleName.grid(row=0, column=0)

	titleProbabilidad = PhotoImage(file = "presentation\\images\\probabilidad100.png")
	lblTitleProb = Label(lblCenterWidgets, image=titleProbabilidad, bg="#20A099")
	lblTitleProb.image = titleProbabilidad
	lblTitleProb.grid(row=0, column=1)

	titleLink = PhotoImage(file = "presentation\\images\\link.png")
	lblTitleLink = Label(lblCenterWidgets, image=titleLink, bg="#20A099")
	lblTitleLink.image = titleLink
	lblTitleLink.grid(row=0, column=2)

	lblAstraName = Label(lblCenterWidgets, text="Astra", bg="#20A099", font=("Arial", 20)).grid(row=1, column=0, pady=10)
	lblAstraProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(0)*100, 1), bg="#20A099", font=("Arial", 20)).grid(row=1, column=1, padx=20, pady=10)
	lblAstraLink = Label(lblCenterWidgets, text="Astra - OsseoSpeed EV", bg="#20A099", font=("Arial", 20), cursor="hand2")
	lblAstraLink.grid(row=1, column=2, padx=20, pady=10)
	lblAstraLink.bind("<Button-1>", lambda event:callback(event, "Astra"))

	lblNobelActiveName = Label(lblCenterWidgets, text="Nobel Active", bg="#20A099", font=("Arial", 20)).grid(row=2, column=0, pady=10)
	lblNobelActiveProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(1)*100, 1), bg="#20A099", font=("Arial", 20)).grid(row=2, column=1, padx=20, pady=10)
	lblNobelActiveLink = Label(lblCenterWidgets, text="Nobel - Active", bg="#20A099", font=("Arial", 20), cursor="hand2")
	lblNobelActiveLink.grid(row=2, column=2, padx=20, pady=10)
	lblNobelActiveLink.bind("<Button-1>", lambda event:callback(event, "NobelActive"))

	lblNobelParallelName = Label(lblCenterWidgets, text="Nobel Parallel", bg="#20A099", font=("Arial", 20)).grid(row=3, column=0, pady=10)
	lblNobelParallelProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(2)*100, 1), bg="#20A099", font=("Arial", 20)).grid(row=3, column=1, padx=20, pady=10)
	lblNobelParallelLink = Label(lblCenterWidgets, text="Nobel - Parallel CC", bg="#20A099", font=("Arial", 20), cursor="hand2")
	lblNobelParallelLink.grid(row=3, column=2, padx=20, pady=10)
	lblNobelParallelLink.bind("<Button-1>", lambda event:callback(event, "NobelParallel"))

	lblNobelReplaceSelectName = Label(lblCenterWidgets, text="Nobel Replace Select", bg="#20A099", font=("Arial", 20)).grid(row=4, column=0, pady=10)
	lblNobelReplaceSelectProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(3)*100, 1), bg="#20A099", font=("Arial", 20)).grid(row=4, column=1, padx=20, pady=10)
	lblNobelReplaceSelectLink = Label(lblCenterWidgets, text="Nobel - Replace Select TC", bg="#20A099", font=("Arial", 20), cursor="hand2")
	lblNobelReplaceSelectLink.grid(row=4, column=2, padx=20, pady=10)
	lblNobelReplaceSelectLink.bind("<Button-1>", lambda event:callback(event, "NobelReplaceSelect"))

	lblNobelReplaceTaperedName = Label(lblCenterWidgets, text="Nobel Replace Tapered", bg="#20A099", font=("Arial", 20)).grid(row=5, column=0, pady=10)
	lblNobelReplaceTaperedProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(4)*100, 1), bg="#20A099", font=("Arial", 20)).grid(row=5, column=1, padx=20, pady=10)
	lblNobelReplaceTaperedLink = Label(lblCenterWidgets, text="Nobel - Replace Tapered", bg="#20A099", font=("Arial", 20), cursor="hand2")
	lblNobelReplaceTaperedLink.grid(row=5, column=2, padx=20, pady=10)
	lblNobelReplaceTaperedLink.bind("<Button-1>", lambda event:callback(event, "NobelReplaceTapered"))

	lblNobelSpeedyReplaceTrichannelName = Label(lblCenterWidgets, text="Nobel Speedy Replace Trichannel", bg="#20A099", font=("Arial", 20)).grid(row=6, column=0, pady=10)
	lblNobelSpeedyReplaceTrichannelProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(5)*100, 1), bg="#20A099", font=("Arial", 20)).grid(row=6, column=1, padx=20, pady=10)
	lblNobelSpeedyReplaceTrichannelLink = Label(lblCenterWidgets, text="Nobel - Speedy Replace TriChannel", bg="#20A099", font=("Arial", 20), cursor="hand2")
	lblNobelSpeedyReplaceTrichannelLink.grid(row=6, column=2, padx=20, pady=10)
	lblNobelSpeedyReplaceTrichannelLink.bind("<Button-1>", lambda event:callback(event, "NobelSpeedyReplaceTrichannel"))

	btnMainMenu = Button(lblBackground, text="Inicio", font=20, command=lambda:buttonMainMenuPressed())
	btnMainMenu.pack(side=TOP, pady=20)

def showFramePredictionResult():
	framePredictionResult.pack(fill=BOTH, expand=True)

def hideFramePredictionResult():
	# Esconder pack_forget() o grid_forget()
	framePredictionResult.pack_forget()

def destroyFramePredictionResult():
	framePredictionResult.destroy()



def main():
	initializeRoot()
	initializeFrameMainMenu()
	showFrameMainMenu()
	root.mainloop()

if __name__ == '__main__':
	root = Tk()
	frameMainMenu = Frame(root)
	frameImageSelected = Frame(root)
	framePredictionResult = Frame(root)
	backgroundImage = PhotoImage(file = "presentation\\images\\background.png")
	# iconSearchImplant = PhotoImage(file = "presentation\\images\\icon_implant.png")
	# titleLogo = PhotoImage(file = "presentation\\images\\logo.png")
	# titleSeleccionarRadiografia = PhotoImage(file = "presentation\\images\\seleccionar_radiografia.png")
	titleResultadosPrediccion = PhotoImage(file = "presentation\\images\\resultados_prediccion.png")
	# titleModelo = PhotoImage(file = "presentation\\images\\modelo.png")
	# titleProbabilidad = PhotoImage(file = "presentation\\images\\probabilidad.png")
	# titleLink = PhotoImage(file = "presentation\\images\\link.png")
	main()
