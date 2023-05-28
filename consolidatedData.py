import os
import cv2
import numpy as np
import pickle

image_dic = os.path.join(os.getcwd(),"images")

images = []
labels = []
for i in os.listdir(image_dic):
    image = cv2.imread(os.path.join(image_dic,i))
    images.append(image)
    label = i.split("_")[0]
    labels.append(label)


images = np.array(images)
labels = np.array(labels)

with open(os.path.join(os.getcwd(),"images.pickle"), "wb") as f:
    pickle.dump(images, f)
with open(os.path.join(os.getcwd(),"labels.pickle"), "wb") as f:
    pickle.dump(labels, f)