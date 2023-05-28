import urllib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

url = "http://192.168.213.2:8080/shot.jpg"
model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

data = []

while len(data)<100:
    img_from_url = urllib.request.urlopen(url)
    image = img_from_url.read()
    # read image as an numpy array
    image = np.array(bytearray(image),np.uint8)
    # use imdecode function
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    facepoints = model.detectMultiScale(image,scaleFactor=2.5)
    if len(facepoints)>0:
        for x,y,w,h in facepoints:
            data.append(image[y-50:y+h+50,x-50:x+w+50])        
    else:
        pass
    cv2.putText(image,f"{len(data)} stored.....",
                (960,50),cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,0,255),thickness=5)
    cv2.imshow("win",image)
    if cv2.waitKey(30)==ord("q"):
        break
cv2.destroyAllWindows()



if len(data)>=100:
    name = input("Enter your name : ")
    for i in range(len(data)):
        cv2.imwrite(os.getcwd() + "/images/" + name + "_" + str(i)+".jpg" , data[i])
    
else:
    print("need more data")
    
    


cv2.imshow("win",data[50])
cv2.waitKey(0)
cv2.destroyAllWindows()
