from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image



def cropImage(file):
	# Read image
	im = cv2.imread(file)
	 
	# Select ROI
	r = cv2.selectROI("image", im)
	
	 
	# Crop image
	imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

	# Display cropped image
	cv2.imshow("Image", imCrop)
	cv2.waitKey(0)


def openFile():
	file = filedialog.askopenfilename(title="Abrir", initialdir="c:", filetypes=(("Ficheros PNG", "*.png"), ("Ficheros JPG", "*.jpg"), ("Todos los ficheros", "*.*")))

	print(file)

	return file


root = Tk()


# root.attributes('-fullscreen', True)
root.state('zoomed')

# Lo del config se puede poner directament en el constructor
root.config(bg="blue")

frameMainMenu = Frame(root)

frameMainMenu.pack(fill=BOTH, expand=1)

# filename = PhotoImage(file = "images\\background.png")
# background_label = Label(frameMainMenu, image=filename)
# background_label.pack(fill=BOTH, expand=1)

icon = PhotoImage(file = "images\\icon_implant.png")
buttonPredict = Button(frameMainMenu, width=200, height=200, image=icon).place(relx=.5, rely=.5, anchor=CENTER)


# Esconder pack_forget() o grid_forget()
# frameMainMenu.pack_forget()





root.mainloop()