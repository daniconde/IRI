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
import random

def openFile():
	file = filedialog.askopenfilename(title="Abrir", initialdir="c:", filetypes=(("Ficheros PNG", "*.png"), ("Ficheros JPG", "*.jpg"), ("Todos los ficheros", "*.*")))

	return file



def initializeRoot():
	root.state('zoomed')

	root.config(bg="blue")

	root.title("Implant Radiography Identifier")

	root.iconbitmap("presentation\\images\\icono.ico")



def buttonMakePredictionPressed():
	destroyFrameMainMenu()
	global frameImageSelected
	frameImageSelected = Frame(root)
	initializeFrameImageSelected("...", None)
	showFrameImageSelected()

def buttonAddRadiographyPressed():
	destroyFrameMainMenu()
	global frameSelectModel
	frameSelectModel = Frame(root)
	initializeFrameSelectModel()
	showFrameSelectModel()

def initializeFrameMainMenu():
	background = PhotoImage(file = "presentation\\images\\background_title.png")
	lblBackground = Label(frameMainMenu, image=background)
	lblBackground.image = background
	lblBackground.pack(fill=BOTH, expand=True)
	
	imageBtnMakePrediction = PhotoImage(file = "presentation\\images\\predecir_implante.png")
	btnMakePrediction = Button(lblBackground, border="0", width=396, height=69, image=imageBtnMakePrediction, command=lambda:buttonMakePredictionPressed())
	btnMakePrediction.image = imageBtnMakePrediction
	btnMakePrediction.place(relx=.5, rely=.45, anchor=CENTER)

	imageBtnAddRadiography = PhotoImage(file = "presentation\\images\\anadir_radiografia.png")
	btnAddRadiography = Button(lblBackground, border="0", width=396, height=69, image=imageBtnAddRadiography, command=lambda:buttonAddRadiographyPressed())
	btnAddRadiography.image = imageBtnAddRadiography
	btnAddRadiography.place(relx=.5, rely=.58, anchor=CENTER)
	
def showFrameMainMenu():
	frameMainMenu.pack(fill=BOTH, expand=True)

def hideFrameMainMenu():
	frameMainMenu.pack_forget()

def destroyFrameMainMenu():
	frameMainMenu.destroy()



def buttonBackFrameSelectModelPressed():
	destroyFrameSelectModel()
	global frameMainMenu
	frameMainMenu = Frame(root)
	initializeFrameMainMenu()
	showFrameMainMenu()

def buttonAstraPressed():
	destroyFrameSelectModel()
	global frameAddRadiography
	frameAddRadiography = Frame(root)
	initializeFrameAddRadiography("Astra", "...", None)
	showFrameAddRadiography()

def buttonNobelActivePressed():
	destroyFrameSelectModel()
	global frameAddRadiography
	frameAddRadiography = Frame(root)
	initializeFrameAddRadiography("NobelActive", "...", None)
	showFrameAddRadiography()

def buttonNobelParallelPressed():
	destroyFrameSelectModel()
	global frameAddRadiography
	frameAddRadiography = Frame(root)
	initializeFrameAddRadiography("NobelParallel", "...", None)
	showFrameAddRadiography()

def buttonNobelReplaceSelectPressed():
	destroyFrameSelectModel()
	global frameAddRadiography
	frameAddRadiography = Frame(root)
	initializeFrameAddRadiography("NobelReplaceSelect", "...", None)
	showFrameAddRadiography()

def buttonNobelReplaceTaperedPressed():
	destroyFrameSelectModel()
	global frameAddRadiography
	frameAddRadiography = Frame(root)
	initializeFrameAddRadiography("NobelReplaceTapered", "...", None)
	showFrameAddRadiography()

def buttonNobelSpeedyReplaceTrichannelPressed():
	destroyFrameSelectModel()
	global frameAddRadiography
	frameAddRadiography = Frame(root)
	initializeFrameAddRadiography("NobelSpeedyReplaceTrichannel", "...", None)
	showFrameAddRadiography()

def initializeFrameSelectModel():
	background = PhotoImage(file = "presentation\\images\\background_deg_selec_modelo.png")
	lblBackground = Label(frameSelectModel, image=background)
	lblBackground.image = background
	lblBackground.pack(fill=BOTH, expand=True)

	iconBack = PhotoImage(file = "presentation\\images\\left_arrow.png")
	btnBack = Button(lblBackground, image=iconBack, bg="#31A190", border="0", width=90, height=90, command=lambda:buttonBackFrameSelectModelPressed())
	btnBack.image = iconBack
	btnBack.pack(side=TOP, anchor=W, padx=10, pady=120)
	
	imageBtnAstra = PhotoImage(file = "presentation\\images\\button_astra.png")
	btnAstra = Button(lblBackground, border="0", width=520, height=55, image=imageBtnAstra, command=lambda:buttonAstraPressed())
	btnAstra.image = imageBtnAstra
	btnAstra.place(relx=.5, rely=.30, anchor=CENTER)

	imageBtnNobelActive = PhotoImage(file = "presentation\\images\\button_nobel_active.png")
	btnNobelActive = Button(lblBackground, border="0", width=520, height=55, image=imageBtnNobelActive, command=lambda:buttonNobelActivePressed())
	btnNobelActive.image = imageBtnNobelActive
	btnNobelActive.place(relx=.5, rely=.40, anchor=CENTER)

	imageBtnNobelParallel = PhotoImage(file = "presentation\\images\\button_nobel_parallel.png")
	btnNobelParallel = Button(lblBackground, border="0", width=520, height=55, image=imageBtnNobelParallel, command=lambda:buttonNobelParallelPressed())
	btnNobelParallel.image = imageBtnNobelParallel
	btnNobelParallel.place(relx=.5, rely=.50, anchor=CENTER)

	imageBtnNobelReplaceSelect = PhotoImage(file = "presentation\\images\\button_nobel_replace_select.png")
	btnNobelReplaceSelect = Button(lblBackground, border="0", width=520, height=55, image=imageBtnNobelReplaceSelect, command=lambda:buttonNobelReplaceSelectPressed())
	btnNobelReplaceSelect.image = imageBtnNobelReplaceSelect
	btnNobelReplaceSelect.place(relx=.5, rely=.60, anchor=CENTER)

	imageBtnNobelReplaceTapered = PhotoImage(file = "presentation\\images\\button_nobel_replace_tapered.png")
	btnNobelReplaceTapered = Button(lblBackground, border="0", width=520, height=55, image=imageBtnNobelReplaceTapered, command=lambda:buttonNobelReplaceTaperedPressed())
	btnNobelReplaceTapered.image = imageBtnNobelReplaceTapered
	btnNobelReplaceTapered.place(relx=.5, rely=.70, anchor=CENTER)

	imageBtnNobelSpeedyReplaceTrichannel = PhotoImage(file = "presentation\\images\\button_nobel_speedy_replace_trichannel.png")
	btnNobelSpeedyReplaceTrichannel = Button(lblBackground, border="0", width=520, height=55, image=imageBtnNobelSpeedyReplaceTrichannel, command=lambda:buttonNobelSpeedyReplaceTrichannelPressed())
	btnNobelSpeedyReplaceTrichannel.image = imageBtnNobelSpeedyReplaceTrichannel
	btnNobelSpeedyReplaceTrichannel.place(relx=.5, rely=.80, anchor=CENTER)
	
def showFrameSelectModel():
	frameSelectModel.pack(fill=BOTH, expand=True)

def hideFrameSelectModel():
	frameSelectModel.pack_forget()

def destroyFrameSelectModel():
	frameSelectModel.destroy()



def buttonBackFrameAddRadiographyPressed():
	destroyFrameAddRadiography()
	global frameSelectModel
	frameSelectModel = Frame(root)
	initializeFrameSelectModel()
	showFrameSelectModel()

def buttonOpenFile2Pressed(model):
	file = openFile()
	photo = PhotoImage(file=file)
	destroyFrameAddRadiography()
	global frameAddRadiography
	frameAddRadiography = Frame(root)
	if file == "":
		initializeFrameAddRadiography(model, "...", None)
	else:
		initializeFrameAddRadiography(model, file, photo)
	showFrameAddRadiography()

def buttonSavePressed(model, file):
	path = ""
	x = random.randint(115,100000)
	if model == "Astra":
		path = "dataset\\INPUT\\ASTRA\\Astra"
	elif model == "NobelActive":
		path = "dataset\\INPUT\\NOBEL_ACTIVE\\nobel_active"
	elif model == "NobelParallel":
		path = "dataset\\INPUT\\NOBEL_PARALLEL_CC\\nobel_parallel"
	elif model == "NobelReplaceSelect":
		path = "dataset\\INPUT\\NOBEL_REPLACE_SELECT\\nobel_replace_select"
	elif model == "NobelReplaceTapered":
		path = "dataset\\INPUT\\NOBEL_REPLACE_TAPERED\\nobel_replace_tapered"
	else:
		path = "dataset\\INPUT\\NOBEL_SPEEDY_REPLACE_TRICHANNEL\\nobel_speedy"

	path = path + str(x) + ".png"

	image = cv2.imread(file)

	cv2.imwrite(path, image)

	destroyFrameAddRadiography()
	global frameMainMenu
	frameMainMenu = Frame(root)
	initializeFrameMainMenu()
	showFrameMainMenu()

def initializeFrameAddRadiography(model, text, photo):
	background = PhotoImage(file = "presentation\\images\\background_deg_anadir_radiografia.png")
	lblBackground = Label(frameAddRadiography, image=background)
	lblBackground.image = background
	lblBackground.pack(fill=BOTH, expand=True)

	routeText = StringVar()
	routeText.set(text)

	iconBack = PhotoImage(file = "presentation\\images\\left_arrow.png")
	btnBack = Button(lblBackground, image=iconBack, bg="#31A190", border="0", width=90, height=90, command=lambda:buttonBackFrameAddRadiographyPressed())
	btnBack.image = iconBack
	btnBack.pack(side=TOP, anchor=W, padx=10, pady=120)

	lblTop1 = Label(lblBackground, bg="#31A190")
	lblTop1.pack(side=TOP)

	lblTop2 = Label(lblTop1, bg="#31A190")
	lblTop2.pack(side=TOP)

	lblRight2 = Label(lblTop1, bg="#31A190")
	lblRight2.pack(side=RIGHT, pady=30)

	lblBottom2 = Label(lblTop1, bg="#31A190")
	lblBottom2.pack(side=BOTTOM)

	lblRouteFileSelected = Label(lblTop2, textvariable=routeText, background="white").pack(side=LEFT)

	imageBtnOpenFile = PhotoImage(file = "presentation\\images\\abrir_archivo.png")
	btnOpenFile = Button(lblTop2, image=imageBtnOpenFile, border="0", width=110, height=20, command=lambda:buttonOpenFile2Pressed(model))
	btnOpenFile.image = imageBtnOpenFile
	btnOpenFile.pack(side=LEFT, padx=10)

	if photo != None:
		lblImageSelected = Label(lblBottom2, image=photo, bg="#31A190", width=400, height=300)
		lblImageSelected.image = photo
		lblImageSelected.pack()
	
	imageBtnSave = PhotoImage(file = "presentation\\images\\guardar.png")
	if photo == None:
		btnSave = Button(lblBackground, image=imageBtnSave, border="0", width=148, height=57, state=DISABLED, command=lambda:buttonSavePressed(model, text))
		btnSave.image = imageBtnSave
		btnSave.pack(side=BOTTOM, pady=30)
	else:
		btnSave = Button(lblBackground, image=imageBtnSave, border="0", width=148, height=57, command=lambda:buttonSavePressed(model, text))
		btnSave.image = imageBtnSave
		btnSave.pack(side=BOTTOM, pady=30)
		
def showFrameAddRadiography():
	frameAddRadiography.pack(fill=BOTH, expand=True)

def hideFrameAddRadiography():
	frameAddRadiography.pack_forget()

def destroyFrameAddRadiography():
	frameAddRadiography.destroy()



def buttonBackFrameImageSelectedPressed():
	destroyFrameImageSelected()
	global frameMainMenu
	frameMainMenu = Frame(root)
	initializeFrameMainMenu()
	showFrameMainMenu()

def buttonOpenFilePressed():
	file = openFile()
	photo = PhotoImage(file=file)
	destroyFrameImageSelected()
	global frameImageSelected
	frameImageSelected = Frame(root)
	if file == "":
		initializeFrameImageSelected("...", None)
	else:
		initializeFrameImageSelected(file, photo)
	showFrameImageSelected()

def modelWithHighestProbability(pred):
	model = 0
	prob = 0
	for i in range(len(pred[0])):
		if i == 0:
			prob = pred.item(i)
			model = i
		elif pred.item(i) > prob:
			prob = pred.item(i)
			model = i

	if model == 0:
		return "Astra"
	elif model == 1:
		return "NobelActive"
	elif model == 2:
		return "NobelParallel"
	elif model == 3:
		return "NobelReplaceSelect"
	elif model == 4:
		return "NobelReplaceTapered"
	else:
		return "NobelSpeedyReplaceTrichannel"

def buttonPredictPressed(file):
	image = convertOneImage(file)
	
	pred = makePrediction(image)

	model = modelWithHighestProbability(pred)
	
	destroyFrameImageSelected()
	global framePredictionResult
	framePredictionResult = Frame(root)
	initializeFramePredictionResult(pred, model)
	showFramePredictionResult()

def initializeFrameImageSelected(text, photo):
	background = PhotoImage(file = "presentation\\images\\background_deg_selec_radiografia.png")
	lblBackground = Label(frameImageSelected, image=background)
	lblBackground.image = background
	lblBackground.pack(fill=BOTH, expand=True)
	
	routeText = StringVar()
	routeText.set(text)

	iconBack = PhotoImage(file = "presentation\\images\\left_arrow.png")
	btnBack = Button(lblBackground, image=iconBack, bg="#31A190", border="0", width=90, height=90, command=lambda:buttonBackFrameImageSelectedPressed())
	btnBack.image = iconBack
	btnBack.pack(side=TOP, anchor=W, padx=10, pady=120)

	lblTop1 = Label(lblBackground, bg="#31A190")
	lblTop1.pack(side=TOP)

	lblTop2 = Label(lblTop1, bg="#31A190")
	lblTop2.pack(side=TOP)

	lblBottom2 = Label(lblTop1, bg="#31A190")
	lblBottom2.pack(side=BOTTOM)

	lblRouteFileSelected = Label(lblTop2, textvariable=routeText, background="white").pack(side=LEFT)

	imageBtnOpenFile = PhotoImage(file = "presentation\\images\\abrir_archivo.png")
	btnOpenFile = Button(lblTop2, image=imageBtnOpenFile, border="0", width=110, height=20, command=lambda:buttonOpenFilePressed())
	btnOpenFile.image = imageBtnOpenFile
	btnOpenFile.pack(side=LEFT, padx=10)

	if photo != None:
		lblImageSelected = Label(lblBottom2, image=photo, bg="#31A190", width=400, height=300)
		lblImageSelected.image = photo
		lblImageSelected.pack()
	
	imageBtnPredict = PhotoImage(file = "presentation\\images\\realizar_prediccion.png")
	if photo == None:
		btnPredict = Button(lblBackground, image=imageBtnPredict, border="0", width=299, height=63, state=DISABLED, command=lambda:buttonPredictPressed(text))
		btnPredict.image = imageBtnPredict
		btnPredict.pack(side=BOTTOM, pady=30)
	else:
		btnPredict = Button(lblBackground, image=imageBtnPredict, border="0", width=299, height=63, command=lambda:buttonPredictPressed(text))
		btnPredict.image = imageBtnPredict
		btnPredict.pack(side=BOTTOM, pady=30)
		

def showFrameImageSelected():
	frameImageSelected.pack(fill=BOTH, expand=True)

def hideFrameImageSelected():
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

def initializeFramePredictionResult(pred, model):
	background = PhotoImage(file = "presentation\\images\\background_deg_result_prediccion2.png")
	lblBackground = Label(framePredictionResult, image=background)
	lblBackground.image = background
	lblBackground.pack(fill=BOTH, expand=True)


	iconBack = PhotoImage(file = "presentation\\images\\left_arrow.png")
	btnBack = Button(lblBackground, image=iconBack, bg="#31A190", border="0", width=90, height=90, command=lambda:buttonBackFramePredictionResult())
	btnBack.image = iconBack
	btnBack.pack(side=TOP, anchor=W, padx=10, pady=120)

	lblCenterWidgets = Label(lblBackground, bg="#31A190")
	lblCenterWidgets.pack(side=TOP)

	if model == "Astra":
		lblAstraName = Label(lblCenterWidgets, text="Astra", bg="#31A190", fg="red", font=("Arial", 20)).grid(row=0, column=0)
		lblAstraProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(0)*100, 1), bg="#31A190", fg="red", font=("Arial", 20)).grid(row=0, column=1, padx=120)
		lblAstraLink = Label(lblCenterWidgets, text="Astra - OsseoSpeed EV", bg="#31A190", fg="red", font=("Arial", 20), cursor="hand2")
		lblAstraLink.grid(row=0, column=2, padx=20)
		lblAstraLink.bind("<Button-1>", lambda event:callback(event, "Astra"))
	else:
		lblAstraName = Label(lblCenterWidgets, text="Astra", bg="#31A190", font=("Arial", 20)).grid(row=0, column=0)
		lblAstraProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(0)*100, 1), bg="#31A190", font=("Arial", 20)).grid(row=0, column=1, padx=120)
		lblAstraLink = Label(lblCenterWidgets, text="Astra - OsseoSpeed EV", bg="#31A190", font=("Arial", 20), cursor="hand2")
		lblAstraLink.grid(row=0, column=2, padx=20)
		lblAstraLink.bind("<Button-1>", lambda event:callback(event, "Astra"))

	if model == "NobelActive":
		lblNobelActiveName = Label(lblCenterWidgets, text="Nobel Active", bg="#31A190", fg="red", font=("Arial", 20)).grid(row=1, column=0, pady=10)
		lblNobelActiveProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(1)*100, 1), bg="#31A190", fg="red", font=("Arial", 20)).grid(row=1, column=1, padx=120, pady=10)
		lblNobelActiveLink = Label(lblCenterWidgets, text="Nobel - Active", bg="#31A190", fg="red", font=("Arial", 20), cursor="hand2")
		lblNobelActiveLink.grid(row=1, column=2, padx=20, pady=10)
		lblNobelActiveLink.bind("<Button-1>", lambda event:callback(event, "NobelActive"))
	else:
		lblNobelActiveName = Label(lblCenterWidgets, text="Nobel Active", bg="#31A190", font=("Arial", 20)).grid(row=1, column=0, pady=10)
		lblNobelActiveProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(1)*100, 1), bg="#31A190", font=("Arial", 20)).grid(row=1, column=1, padx=120, pady=10)
		lblNobelActiveLink = Label(lblCenterWidgets, text="Nobel - Active", bg="#31A190", font=("Arial", 20), cursor="hand2")
		lblNobelActiveLink.grid(row=1, column=2, padx=20, pady=10)
		lblNobelActiveLink.bind("<Button-1>", lambda event:callback(event, "NobelActive"))

	if model == "NobelParallel": 
		lblNobelParallelName = Label(lblCenterWidgets, text="Nobel Parallel", bg="#31A190", fg="red", font=("Arial", 20)).grid(row=2, column=0, pady=10)
		lblNobelParallelProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(2)*100, 1), bg="#31A190", fg="red", font=("Arial", 20)).grid(row=2, column=1, padx=120, pady=10)
		lblNobelParallelLink = Label(lblCenterWidgets, text="Nobel - Parallel CC", bg="#31A190", fg="red", font=("Arial", 20), cursor="hand2")
		lblNobelParallelLink.grid(row=2, column=2, padx=20, pady=10)
		lblNobelParallelLink.bind("<Button-1>", lambda event:callback(event, "NobelParallel"))
	else:
		lblNobelParallelName = Label(lblCenterWidgets, text="Nobel Parallel", bg="#31A190", font=("Arial", 20)).grid(row=2, column=0, pady=10)
		lblNobelParallelProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(2)*100, 1), bg="#31A190", font=("Arial", 20)).grid(row=2, column=1, padx=120, pady=10)
		lblNobelParallelLink = Label(lblCenterWidgets, text="Nobel - Parallel CC", bg="#31A190", font=("Arial", 20), cursor="hand2")
		lblNobelParallelLink.grid(row=2, column=2, padx=20, pady=10)
		lblNobelParallelLink.bind("<Button-1>", lambda event:callback(event, "NobelParallel"))

	if model == "NobelReplaceSelect": 
		lblNobelReplaceSelectName = Label(lblCenterWidgets, text="Nobel Replace Select", bg="#31A190", fg="red", font=("Arial", 20)).grid(row=3, column=0, pady=10)
		lblNobelReplaceSelectProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(3)*100, 1), bg="#31A190", fg="red", font=("Arial", 20)).grid(row=3, column=1, padx=120, pady=10)
		lblNobelReplaceSelectLink = Label(lblCenterWidgets, text="Nobel - Replace Select TC", bg="#31A190", fg="red", font=("Arial", 20), cursor="hand2")
		lblNobelReplaceSelectLink.grid(row=3, column=2, padx=20, pady=10)
		lblNobelReplaceSelectLink.bind("<Button-1>", lambda event:callback(event, "NobelReplaceSelect"))
	else:
		lblNobelReplaceSelectName = Label(lblCenterWidgets, text="Nobel Replace Select", bg="#31A190", font=("Arial", 20)).grid(row=3, column=0, pady=10)
		lblNobelReplaceSelectProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(3)*100, 1), bg="#31A190", font=("Arial", 20)).grid(row=3, column=1, padx=120, pady=10)
		lblNobelReplaceSelectLink = Label(lblCenterWidgets, text="Nobel - Replace Select TC", bg="#31A190", font=("Arial", 20), cursor="hand2")
		lblNobelReplaceSelectLink.grid(row=3, column=2, padx=20, pady=10)
		lblNobelReplaceSelectLink.bind("<Button-1>", lambda event:callback(event, "NobelReplaceSelect"))

	if model == "NobelReplaceTapered":
		lblNobelReplaceTaperedName = Label(lblCenterWidgets, text="Nobel Replace Tapered", bg="#31A190", fg="red", font=("Arial", 20)).grid(row=4, column=0, pady=10)
		lblNobelReplaceTaperedProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(4)*100, 1), bg="#31A190", fg="red", font=("Arial", 20)).grid(row=4, column=1, padx=120, pady=10)
		lblNobelReplaceTaperedLink = Label(lblCenterWidgets, text="Nobel - Replace Tapered", bg="#31A190", fg="red", font=("Arial", 20), cursor="hand2")
		lblNobelReplaceTaperedLink.grid(row=4, column=2, padx=20, pady=10)
		lblNobelReplaceTaperedLink.bind("<Button-1>", lambda event:callback(event, "NobelReplaceTapered"))
	else:
		blNobelReplaceTaperedName = Label(lblCenterWidgets, text="Nobel Replace Tapered", bg="#31A190", font=("Arial", 20)).grid(row=4, column=0, pady=10)
		lblNobelReplaceTaperedProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(4)*100, 1), bg="#31A190", font=("Arial", 20)).grid(row=4, column=1, padx=120, pady=10)
		lblNobelReplaceTaperedLink = Label(lblCenterWidgets, text="Nobel - Replace Tapered", bg="#31A190", font=("Arial", 20), cursor="hand2")
		lblNobelReplaceTaperedLink.grid(row=4, column=2, padx=20, pady=10)
		lblNobelReplaceTaperedLink.bind("<Button-1>", lambda event:callback(event, "NobelReplaceTapered"))

	if model == "NobelSpeedyReplaceTrichannel":
		lblNobelSpeedyReplaceTrichannelName = Label(lblCenterWidgets, text="Nobel Speedy Replace Trichannel", bg="#31A190", fg="red", font=("Arial", 20)).grid(row=5, column=0, pady=10)
		lblNobelSpeedyReplaceTrichannelProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(5)*100, 1), bg="#31A190", fg="red", font=("Arial", 20)).grid(row=5, column=1, padx=120, pady=10)
		lblNobelSpeedyReplaceTrichannelLink = Label(lblCenterWidgets, text="Nobel - Speedy Replace TriChannel", bg="#31A190", fg="red", font=("Arial", 20), cursor="hand2")
		lblNobelSpeedyReplaceTrichannelLink.grid(row=5, column=2, padx=20, pady=10)
		lblNobelSpeedyReplaceTrichannelLink.bind("<Button-1>", lambda event:callback(event, "NobelSpeedyReplaceTrichannel"))
	else:
		lblNobelSpeedyReplaceTrichannelName = Label(lblCenterWidgets, text="Nobel Speedy Replace Trichannel", bg="#31A190", font=("Arial", 20)).grid(row=5, column=0, pady=10)
		lblNobelSpeedyReplaceTrichannelProb = Label(lblCenterWidgets, text=locale.format("%f", pred.item(5)*100, 1), bg="#31A190", font=("Arial", 20)).grid(row=5, column=1, padx=120, pady=10)
		lblNobelSpeedyReplaceTrichannelLink = Label(lblCenterWidgets, text="Nobel - Speedy Replace TriChannel", bg="#31A190", font=("Arial", 20), cursor="hand2")
		lblNobelSpeedyReplaceTrichannelLink.grid(row=5, column=2, padx=20, pady=10)
		lblNobelSpeedyReplaceTrichannelLink.bind("<Button-1>", lambda event:callback(event, "NobelSpeedyReplaceTrichannel"))

	imageBtnMainMenu = PhotoImage(file = "presentation\\images\\inicio.png")
	btnMainMenu = Button(lblBackground, image=imageBtnMainMenu, border="0", width=102, height=56, command=lambda:buttonMainMenuPressed())
	btnMainMenu.image = imageBtnMainMenu
	btnMainMenu.pack(side=TOP, pady=40)

def showFramePredictionResult():
	framePredictionResult.pack(fill=BOTH, expand=True)

def hideFramePredictionResult():
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
	frameSelectModel = Frame(root)
	frameAddRadiography = Frame(root)
	frameImageSelected = Frame(root)
	framePredictionResult = Frame(root)
	main()
