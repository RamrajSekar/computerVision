import cv2
import matplotlib.pyplot as plt
import winsound

#Using Opencv predefined model based on Tensorflow: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
cofig_file = "securityCam/resources/configFiles/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "securityCam/resources/configFiles/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb"
img_labels = "securityCam/resources/otherFiles/Labels.txt"
img_file = "securityCam/resources/otherFiles/image.jpg"
plot_imgFile = "securityCam/resources/otherFiles/"
font_scale = 3
font = cv2.FONT_HERSHEY_COMPLEX
model = cv2.dnn_DetectionModel(frozen_model,cofig_file)
# Below is the lables of names to confirm from the image
class_imgLabels = []
with open(img_labels,'rt') as imgLabels:
    class_imgLabels = imgLabels.read().rstrip('\n').split('\n')

# def openCamera(camIndex):
#     camera = cv2.VideoCapture(camIndex)
#     while camera.isOpened():
#         #ret refers to retrieve and frame refers to frame of the camera
#         ret, frame = camera.read()
#         #decalre a key to close the camera using ord
#         if cv2.waitKey(10) == ord('q'):
#             print("Closing the webcam")
#             break
#         #Name of the camera
#         cv2.imshow("My Camera", frame)
def imgClassify(fileName):
    
    model.setInputSize(320,320)
    model.setInputScale(1.0/127.5)
    model.setInputMean(127.5)
    model.setInputSwapRB(True) #RGB will be converted to grey automatically
    img = cv2.imread(fileName)
    # print(fileName)
    plt.imshow(img)
    classIndex, confidece, bbox = model.detect(img,confThreshold=0.55)
    print(classIndex)

    for classInd, conf, boxes in zip(classIndex.flatten(),confidece.flatten(),bbox):
        cv2.rectangle(img,boxes,(255,0,0),2)
        cv2.putText(img,class_imgLabels[classInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)
    # plt.imshow(img)
    plt.savefig(plot_imgFile+'newImg.png')

def webCamCapture(camIndex):
    camera = cv2.VideoCapture(camIndex)
    camera.set(3,640)
    camera.set(4,480)
    camera.set(10,100)
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
            winsound.PlaySound('securityCam/resources/otherFiles/alert.wav',winsound.SND_ASYNC)
        #decalre a key to close the camera using ord
        if cv2.waitKey(10) == ord('q'):
            print("Closing the webcam")
            break
        # print(threshold)
        # print(type(threshold))
        # classIndex, confidece, bbox = model.detect(frame1,confThreshold=0.5)
        # if len(classIndex)!=0:
        #     for classInd, conf, boxes in zip(classIndex.flatten(),confidece.flatten(),bbox):
        #         if classInd<=80:
        #             cv2.rectangle(frame1,boxes,(255,0,0),2)
        #             cv2.putText(frame1,class_imgLabels[classInd-1].upper(),(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)
        #Name of the camera
        cv2.imshow("My Camera", frame1)


def objReadFrmCam(camIndex):
    model.setInputSize(320,320)
    model.setInputScale(1.0/127.5)
    model.setInputMean(127.5)
    model.setInputSwapRB(True)
    camera = cv2.VideoCapture(camIndex)
    ret, frame = camera.read()
    while camera.isOpened():
        ret, frame = camera.read()
        if cv2.waitKey(10) == ord('q'):
            print("Closing the webcam")
            break
        else:
            classIndex, confidece, bbox = model.detect(frame,confThreshold=0.5)
            if len(classIndex)!=0:
                for classInd, conf, boxes in zip(classIndex.flatten(),confidece.flatten(),bbox):
                    if classInd<=80:
                        cv2.rectangle(frame,boxes,(255,0,0),2)
                        cv2.putText(frame,class_imgLabels[classInd-1].upper(),(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)
            #Name of the camera
            cv2.imshow("My Camera", frame)


if __name__ == '__main__':
# openCamera(0)
    # webCamCapture(0)
    # print(class_imgLabels)
    # imgClassify(img_file)
    objReadFrmCam(0)