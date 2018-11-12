import cv2
import os

def load_images_from_folder(folder):
    images = []
    i=0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            imgpq = cv2.resize(img,(32,32))
            file_name = "./F/F" + str(i) + ".jpg"
            cv2.imwrite(file_name, imgpq)
            i+=1

folder = "./F/"
load_images_from_folder(folder)


