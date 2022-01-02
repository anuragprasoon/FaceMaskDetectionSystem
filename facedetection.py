import cv2 as cv
faceCascade = cv.CascadeClassifier('facedetect.xml')
mouthCascade = cv.CascadeClassifier('haarcascade_mcs_mouth.xml')
video_capture = cv.VideoCapture(0)

def resizing(frame,scale=0.50):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dim=(width,height)
 
    return cv.resize(frame,dim,interpolation=cv.INTER_AREA)


while True:
    ret, frame = video_capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=9,minSize=(60, 60),flags=cv.CASCADE_SCALE_IMAGE)
    mouths = mouthCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10,minSize=(60, 60),flags=cv.CASCADE_SCALE_IMAGE)
    if len(faces)==0:
         cv.putText(frame,'NO FACE DETECTED',(10,30),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,0,0),2)
    else:
        cv.putText(frame,'No. of Person in the frame: '+str(len(faces)),(10,30),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,0,0),2)
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0), 2)
        cv.putText(frame,'Face Detected',(x-10,y-10),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),2)
        if len(mouths) !=0:
            for (x,y,w,h) in mouths:
                cv.rectangle(frame, (x,y),(x+w,y+h),(0,0,255), 2)
                cv.putText(frame,'No Mask Detected',(x-10,y-10),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,0,255),2)
                cv.putText(frame,'Pls Wear Mask!',(x-10,y+h+30),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,0,255),2)
        else:
            cv.putText(frame,'Mask Detected!',(x-10,y+h+30),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),2)
            
         # Display the resulting frame
    resized=resizing(frame)
    cv.imshow('Video',resized)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv.destroyAllWindows()
