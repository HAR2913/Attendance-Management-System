import cv2
import face_recognition
 
imgHar = face_recognition.load_image_file('Face-Recognition-master\Image basic\Hardik.jpg')
imgHar = cv2.cvtColor(imgHar,cv2.COLOR_BGR2RGB)
imgKir = face_recognition.load_image_file('Face-Recognition-master\Image basic\Kiresh.jpg')
imgKir = cv2.cvtColor(imgKir,cv2.COLOR_BGR2RGB)
imgDha = face_recognition.load_image_file('Face-Recognition-master\Image basic\Dharaneesh.jpg')
imgDha = cv2.cvtColor(imgDha,cv2.COLOR_BGR2RGB)
imgAsh = face_recognition.load_image_file('Face-Recognition-master\Image basic\Ashwin.jpg')
imgAsh = cv2.cvtColor(imgAsh,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgHar)[0]
encodeHar = face_recognition.face_encodings(imgHar)[0]
cv2.rectangle(imgHar,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoc = face_recognition.face_locations(imgDha)[0]
encodeDha = face_recognition.face_encodings(imgDha)[0]
cv2.rectangle(imgDha,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoc = face_recognition.face_locations(imgAsh)[0]
encodeAsh = face_recognition.face_encodings(imgAsh)[0]
cv2.rectangle(imgAsh,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
 
faceLocTest = face_recognition.face_locations(imgKir)[0]
encodeTest = face_recognition.face_encodings(imgKir)[0]
cv2.rectangle(imgKir,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
 
results = face_recognition.compare_faces([encodeHar],encodeTest)
faceDis = face_recognition.face_distance([encodeHar],encodeTest)
print(results,faceDis)
cv2.putText(imgKir,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
cv2.imshow('Hardik',imgHar)
cv2.imshow('kiresh',imgKir)
cv2.waitKey(0)
