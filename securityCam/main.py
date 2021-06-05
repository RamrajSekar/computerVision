import cv2
import winsound

# print(camera.isOpened())/

def openCamera(camIndex):
    camera = cv2.VideoCapture(camIndex)
    while camera.isOpened():
        #ret refers to retrieve and frame refers to frame of the camera
        ret, frame = camera.read()
        #decalre a key to close the camera using ord
        if cv2.waitKey(10) == ord('q'):
            print("Closing the webcam")
            break
        #Name of the camera
        cv2.imshow("My Camera", frame)

def getFrameDiff(camIndex):
    camera = cv2.VideoCapture(camIndex)
    while camera.isOpened():
        #ret refers to retrieve and frame refers to frame of the camera
        ret, frame1 = camera.read()
        ret, frame2 = camera.read()
        #absdif is not recommended
        frameDiff = cv2.absdiff(frame1,frame2)
        #so convert color using below
        converToGrey = cv2.cvtColor(frameDiff,cv2.COLOR_RGB2GRAY)
        #Convert to blur image and kernal size and sigma
        converToblur = cv2.GaussianBlur(converToGrey,(5,5),0)
        # Threshold is used to get rid of the noise/unwanted things and convert the img to little bit sharper
        _, threshold = cv2.threshold(converToblur,20,255,cv2.THRESH_BINARY)
        # Once after removing unwanted thing make the things available in camera bigger using dilation
        dilated =cv2.dilate(threshold,None, iterations=3)
        # To declare the border to find the items use Contours where RETR_TREE is the mode
        contours, _ = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # To give border
        # cv2.drawContours(frame1,contours,-1,(0,255,0),2)
        # To highlight the larger items in camera
        for c in contours:
            if cv2.contourArea(c) < 5000:
                continue
            #To get the length,height and wdith
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
            # To play default sound
            # winsound.Beep(500, 200)
            # To play custom sound asyncronously
            winsound.PlaySound('securityCam/alert.wav',winsound.SND_ASYNC)
        #decalre a key to close the camera using ord
        if cv2.waitKey(10) == ord('q'):
            print("Closing the webcam")
            break
        #Name of the camera
        cv2.imshow("My Camera", frame1)

if __name__ == '__main__':
# openCamera(0)
    getFrameDiff(0)