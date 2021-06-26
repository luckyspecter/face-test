import cv2
import numpy as np

"""
cv2中face子模块目前支持的算法有:
        （1）主成分分析（PCA）——Eigenfaces（特征脸）——函数：cv2.face.EigenFaceRecognizer_create()
PCA：低维子空间是使用主元分析找到的，找具有最大方差的哪个轴。
缺点：若变化基于外部（光照），最大方差轴不一定包括鉴别信息，不能实行分类。
        （2）线性判别分析（LDA）——Fisherfaces（特征脸）——函数： ：cv2.face.FisherFaceRecognizer_create()
LDA:线性鉴别的特定类投影方法，目标：实现类内方差最小，类间方差最大。
        （3）局部二值模式（LBP）——LocalBinary Patterns Histograms——函数：cv2.face.LBPHFaceRecognizer_create()

"""
recognizer = cv2.face.LBPHFaceRecognizer_create() #人脸识别器这里与训练使用算法一致
recognizer.read('./model/face_model_trained.xml')#引入自己训练好的模型
cascadePath = "./model/haarcascade_frontalface_default.xml"#引入官方模型
faceCascade = cv2.CascadeClassifier(cascadePath)#分级器
font = cv2.FONT_HERSHEY_SIMPLEX

idnum = 0

cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)#打开本地摄像头设备
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img = cam.read()#读取图片
    if ret is None:#没有读到图片
        break
    # cv2.imshow("img",img)#读取的原摄像头的图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#使图片灰度化

    #使用训练好的文件开始检测人脸
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,#缩放比例
        minNeighbors=5,
        minSize=(int(minW), int(minH))#最小识别
    )

    for (x, y, w, h) in faces:#扫描图片
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)#将检测到人脸框起来，绿色RBG=255，然后边框的大小，其他值默认
        idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])#用自己的模型来预测

        #统计黑色概率
        if confidence < 100:
            confidence = "{0}%".format(round(100 - confidence))
        else:
            idnum = "unknown"
            confidence = "{0}%".format(round(100 - confidence))

        cv2.putText(img, str(idnum), (x+5, y-5), font, 1, (0, 0, 255), 1)
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (0, 0, 0), 1)


        roi_color_face=img[y:y+h,x:x+w]#将图片的人脸提取出来
        cv2.imshow("roi_color_face",roi_color_face)
    cv2.imshow('camera', img)#检测到的人脸的图像
    k = cv2.waitKey(25)#设置一个退出标志为ESC每25ms读取一次
    if k == 27:#ESC的ascall为27
        break

cam.release()#释放图片
cv2.destroyAllWindows()#关闭所有窗口
"""
注：1. 11行的names中存储人的名字，若该人id为0则他的名字在第一位，id位1则排在第二位，以此类推。
注：2. 最终效果为一个绿框，框住人脸，左上角为红色的人名，左下角为黑色的概率。
"""