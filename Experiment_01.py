import numpy as np
import cv2
import matplotlib.pyplot as plt

#img path
img = cv2.imread("dogesh.jpg")

#check image load or not
if img is None:
    print("image not load")

else:
    #convert BGR to RGB
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #Original image
    plt.imshow(img_rgb)
    plt.title("Original image")
    plt.axis("off")
    plt.show()

    #Resize the image
    resize = cv2.resize(img,(150,150))
    plt.imshow(resize)
    plt.title("resized image")
    plt.axis("off")
    plt.show()

    #covert in gray scale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    plt.imshow(gray,cmap = "gray")
    plt.title("gray scale image")
    plt.axis("off")
    plt.show()

    #output
    cv2.imwrite("This is Gray image",gray)

    