import os
import numpy as np
import cv2
import sys
# pick up and save image from video
if __name__ == '__main__':

    argvs = sys.argv
    if (len(sys.argv) != 2):
        print ("incorrect argument: Please 1 element.")
        sys.exit()
#    if (sys.argv[1].find("mp4") == -1):
 #       print ("incorrect argument: Please only mp4 file.")
  #      sys.exit()

    path = argvs[1]
    #path = path.split(".")[0] + "_60sec"
    path = path.split(".")[0]
    print (path)
    if os.path.exists(path) == False:
        os.mkdir(path)

    cap = cv2.VideoCapture(argvs[1])

    count = 0

    #for count in range(900):
    while (cap.isOpened()):
        print (count)
        ret, img = cap.read()
        if ret==True:
            cv2.imwrite(path + "/frame" + str(count) + ".jpg", img)
            count += 1
        else:
            break

    cap.release()

    #print ("count: ", count) # 60sec = 900frames
