import cv2
import numpy as np
import scipy
import PIL.Image
import math
import time
#from config_reader import config_reader
#import util
import copy
import matplotlib
matplotlib.use('Agg')
#%matplotlib inline
import pylab as plt
import pickle
import sys
import dlib
import os

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]
# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
          [55,56], [37,38], [45,46]]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
cmap = matplotlib.cm.get_cmap('hsv')


def openpose(canvas, fnum):
    all_peaks = entry['all_peaks_frame' + str(fnum) + '.jpg']
    subset = entry['subset_frame' + str(fnum) + '.jpg']
    candidate = entry['candidate_frame' + str(fnum) + '.jpg']
    #test_image = path + '2017-06-29_' + str(time1) + '_2017-06-29_' + str(time2) + '/frame' + str(fnum) + '.jpg'
    #canvas = cv2.imread(test_image) # B,G,R order
    # visualize 2
    stickwidth = 4
    for n in range(len(subset)):
    #for n in range(1,2):
        for i in range(17):
            index = subset[n][np.array(limbSeq[i])-1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            cv2.circle(cur_canvas, (int(Y[0]), int(X[0])), 4, colors[i], thickness=-1)
            cv2.circle(cur_canvas, (int(Y[1]), int(X[1])), 4, colors[i], thickness=-1)
            canvas = cv2.addWeighted(canvas, 0.3, cur_canvas, 0.7, 0)

    return canvas

if __name__ == '__main__':
    start_time = time.time()
    argvs = sys.argv
    if (len(sys.argv) != 3):
        print ('incorrect argument: Please select pickle and mp4')
        sys.exit()

    pckl_path = argvs[1]
    video = argvs[2]  # 83f051ea_2017-08-03_090000_2017-08-03_093000.mp4

    vlist = video.split('_')
    path = vlist[0]
    tlist = vlist[1].split('-')
    year = tlist[0]
    month = tlist[1]
    day = tlist[2]
    time1 = vlist[2]
    time2 = vlist[4].split('.')[0]

    try:
        #with open(path + 'results/2017-06-29_' + str(time1) + '_2017-06-29_' + str(time2) + '_result.pickle', 'rb') as f:
        with open(pckl_path, 'rb') as f:
            entry = pickle.load(f)
    except EOFError:
         print ('error')

    #fcount = len(entry) / 3
    #fcount = int(fcount)

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out = cv2.VideoWriter(path+'_'+year+'-'+month+'-'+day+'_'+time1+'_'+year+'-'+month+'-'+day+'_'+time2+'_result.mp4', fourcc, 15.25, (1280, 720))

    cap = cv2.VideoCapture(video)
    fnum = 0
    while(cap.isOpened()):
        flag, img = cap.read()  # Capture frame-by-frame
        if flag == False:  # Is a frame left?
            break
        else:
            if fnum % 100 = 0:
                print (fnum)
            img = openpose(img, fnum)
            out.write(img)
            fnum += 1
        #if fnum == 10:
        #    break

    cap.release()
    out.release()

    end_time = time.time()
    print ("Finish!")
    print ("elasped time: " + str(end_time - start_time))
