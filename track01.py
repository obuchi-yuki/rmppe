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

# 二つの矩形の面積重なり率
def conarea(reca, recb): # rec=(y,x,h,w)
    ya, xa, ha, wa = reca
    yb, xb, hb, wb = recb
    sreca = ha * wa
    if ya + ha < yb or yb + hb < ya or xa + wa < xb or xb + wb < xa:
        return 0
    else:
        if ya < yb:
            if xa < xb:
                if ya + ha < yb + hb:
                    if xa + wa < xb + wb:
                        return (ya + ha - yb) * (xa + wa - xb) / sreca
                    else:
                        return (ya + ha - yb) * wb / sreca
                else:
                    if xa + wa < xb + wb:
                        return hb * (xa + wa - xb) / sreca
                    else:
                        return hb * wb / sreca

            else:
                if ya + ha < yb + hb:
                    if xa + wa > xb + wb:
                        return (ya + ha - yb) * (xb + wb - xa) / sreca
                    else:
                        return (ya + ha - yb) * wa / sreca
                else:
                    if xa + wa > xb + wb:
                        return hb * (xb + wb - xa) / sreca
                    else:
                        return hb * wa / sreca

        else:
            if xa > xb:
                if ya + ha > yb + hb:
                    if xa + wa > xb + wb:
                        return (yb + hb - ya) * (xb + wb - xa) / sreca
                    else:
                        return (yb + hb - ya) * wa / sreca
                else:
                    if xa + wa > xb + wb:
                        return ha * (xb + wb - xa) / sreca
                    else:
                        return ha * wa / sreca

            else:
                if ya + ha > yb + hb:
                    if xa + wa < xb + wb:
                        return (yb + hb - ya) * (xa + wa - xb) / sreca
                    else:
                        return (yb + hb - ya) * wb / sreca
                else:
                    if xa + wa < xb + wb:
                        return ha * (xa + wa - xb) / sreca
                    else:
                        return ha * wb / sreca

if __name__ == '__main__':
    start_time = time.time()
    argvs = sys.argv
    if (len(sys.argv) != 3):
        print ('incorrect argument: Please select pickle and frame_directory')
        sys.exit()

    pckl_path = argvs[1]
    f_dir = argvs[2]
    #### トラッカー生成
    tracker = dlib.correlation_tracker()
    #resize_rate = 1

    try:
        with open(pckl_path, 'rb') as f:
            entry = pickle.load(f)
    except EOFError:
        print ('pickle_error')


    fcount = int(len(entry) / 3)
    path2 = f_dir  + '_track'
    if os.path.exists(path2) == False:
        os.mkdir(path2)

    # フォント指定
    fontType = cv2.FONT_HERSHEY_SIMPLEX
    stickwidth = 4

    test_image = f_dir + '/frame' + str(0) + '.jpg'
    orgimg = cv2.imread(test_image)
    canvas = cv2.imread(test_image) # B,G,R order
    subset = entry['subset_frame' + str(0) + '.jpg']
    candidate = entry['candidate_frame' + str(0) + '.jpg']

    person_num = len(subset)
    people = []
    mlabels = []
    mlabel = []

    track_windows = []
    roi_hists = []
    term_crits = []


    for n in range(len(subset)):
    #for n in range(1,2):
        #print (n)
        person = []
        xlist = []
        ylist = []
        mlabel.append(n)
        for i in range(17):
            index = subset[n][np.array(limbSeq[i])-1]
            if -1 in index:
                #print (str(i) + '= -1')
                Y = np.array([-1, -1])
                X = np.array([-1, -1])
                person.append([Y, X])
                continue
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            ylist.append(Y[0])
            ylist.append(Y[1])
            xlist.append(X[0])
            xlist.append(X[1])
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])
            cv2.circle(canvas, (int(Y[0]), int(X[0])), 4, colors[i], thickness=-1)
            cv2.circle(canvas, (int(Y[1]), int(X[1])), 4, colors[i], thickness=-1)
            #print ('X: ', X)
            #print ('Y: ', Y)
            person.append([Y, X])

        r = 0.25
        y1 = int(min(ylist) - (max(ylist) - min(ylist)) * r)
        y2 = int(max(ylist) + (max(ylist) - min(ylist)) * r)
        x1 = int(min(xlist) - (max(xlist) - min(xlist)) * r)
        x2 = int(max(xlist) + (max(xlist) - min(xlist)) * r)
        cv2.rectangle(canvas, (y1, x1), (y2, x2), (255, 255, 255), 4)

         # 追跡する枠を決定
        track_window = (y1, x1, y2 - y1, x2 - x1)
        #x, y, w, h = track_window
        #cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 0), 4)
        #roi = orgimg[x1:x2, y1:y2]
        #hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        #img_mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        #roi_hist = cv2.calcHist([hsv_roi], [0], img_mask, [180], [0,180])
        #cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        #term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        track_windows.append(track_window)
        #roi_hists.append(roi_hist)
        #term_crits.append(term_crit)

        cv2.putText(canvas, str(n), (y1 + 5, x1 - 10), fontType, 1, (255, 255, 255), 4)
        people.append(person)

    mlabels.append(mlabel)
    #plt.figure()
    #plt.imshow(canvas[:,:,[2,1,0]])
    #fig = matplotlib.pyplot.gcf()
    #fig.set_size_inches(12, 12)
    #plt.close()
    #canvas.savefig(path2 + '/frame' + str(0) + '.jpg')
    cv2.imwrite(path2 + '/frame' + str(0) + '.jpg', canvas)
    for fnum2 in range(1, fcount):
        test_image = f_dir + '/frame' + str(fnum2) + '.jpg'
        canvas = cv2.imread(test_image) # B,G,R order
        print (str(fnum2 - 1)  + ' & ' + str(fnum2))

        subset2 = entry['subset_frame' + str(fnum2) + '.jpg']
        candidate2 = entry['candidate_frame' + str(fnum2) + '.jpg']


        # 処理負荷軽減のための対象フレーム縮小（引数指定時）
        #height, width = canvas.shape[:2]
        #temp_frame = cv2.resize(canvas, (int(width/resize_rate), int(height/resize_rate)))
        temp_frame = canvas
        track_windows2 = []
        for ti in range(len(track_windows)):
            #hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
            #dst = cv2.calcBackProject([hsv], [0], roi_hists[ti], [0, 180], 1)
            #ret, track_window = cv2.meanShift(dst, track_windows[ti], term_crits[ti])
            y, x, h, w = track_windows[ti]
            #cv2.rectangle(canvas, (y, x), (y + h, x + w), (255, 255, 0), 4)

            tracker.start_track(temp_frame, dlib.rectangle(y, x, y + h, x + w))
            tracker.update(temp_frame)
            tracking_point = tracker.get_position()
            #print ('track: ', tracking_point)
            tracking_point_x1 = int(tracking_point.left())
            tracking_point_y1 = int(tracking_point.top())
            tracking_point_x2 = int(tracking_point.right())
            tracking_point_y2 = int(tracking_point.bottom())
            track_window = (tracking_point_x1, tracking_point_y1, tracking_point_x2 - tracking_point_x1, tracking_point_y2 - tracking_point_y1)
            #cv2.rectangle(canvas, (tracking_point_x1, tracking_point_y1), (tracking_point_x2, tracking_point_y2), (0, 255, 255), 4)
            track_windows2.append(track_window)

        people2 = []
        mlabel2 = []
        subwin = []
        for n in range(len(subset2)):
        #for n in range(1):
            #print (n)
            xysum = []
            #xylists = []
            person = []
            xlist = []
            ylist = []
            for pn in range(len(people)):
                xylen = 0
                xycon = 0
                xylist = []
                for i in range(17):
                    index = subset2[n][np.array(limbSeq[i])-1]
                    if -1 in index:
                        #print (str(i) + '= -1')
                        #xylist.append(np.array([-1, -1]))
                        Y = np.array([-1, -1])
                        X = np.array([-1, -1])
                        if pn == 0:
                            person.append([Y, X])
                        continue

                    Y = candidate2[index.astype(int), 0]
                    X = candidate2[index.astype(int), 1]
                    if pn == 0:
                        person.append([Y, X])

                        ylist.append(Y[0])
                        ylist.append(Y[1])
                        xlist.append(X[0])
                        xlist.append(X[1])
                        mX = np.mean(X)
                        mY = np.mean(Y)
                        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                        polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                        cv2.fillConvexPoly(canvas, polygon, colors[i])
                        cv2.circle(canvas, (int(Y[0]), int(X[0])), 4, colors[i], thickness=-1)
                        cv2.circle(canvas, (int(Y[1]), int(X[1])), 4, colors[i], thickness=-1)
                    #print ('X: ', X)
                    #print ('Y: ', Y)

                    Y0 = people[pn][i][0]
                    X0 = people[pn][i][1]
                    if -1 in Y0:
                        #print (str(i) + '= -1 (0)')
                        #xylist.append(np.array([-2, -2]))
                        continue
                    xycon += 1
                    xylen += (np.sqrt((X[0] - X0[0]) ** 2 + (Y[0] - Y0[0]) ** 2) \
                              + np.sqrt((X[1] - X0[1]) ** 2 + (Y[1] - Y0[1]) ** 2)) / 2
                    #xylist.append(np.array([np.sqrt((X[0] - X0[0]) ** 2 + (Y[0] - Y0[0]) ** 2) \
                    #                   , np.sqrt((X[1] - X0[1]) ** 2 + (Y[1] - Y0[1]) ** 2)]))


                if xycon == 0:
                    continue

                xysum.append(xylen / xycon)
                #xylists.append(xylist)

            people2.append(person)
            pn = np.argsort(xysum)[0]

            r = 0.25
            y1 = int(min(ylist) - (max(ylist) - min(ylist)) * r)
            y2 = int(max(ylist) + (max(ylist) - min(ylist)) * r)
            x1 = int(min(xlist) - (max(xlist) - min(xlist)) * r)
            x2 = int(max(xlist) + (max(xlist) - min(xlist)) * r)
            win = (y1, x1, y2 - y1, x2 - x1)
            subwin.append(win)
            if xysum[pn] < 20:
                #print (str(n) + ' is ' + str(pn) + ' : ' + str(xysum[pn]))
                pnn = mlabel[pn]
                mlabel2.append(pnn)
                track_windows[pnn] = win
            else:
                #print (str(n) + ' is NaN (' + str(pn) + ' : ' + str(xysum[pn]) + ')')
                mlabel2.append(-1)


        for mi in range(len(mlabel2)):
            if mlabel2[mi] == -1:
                c = 0
                cs = []
                for ti in range(len(track_windows)):
                    cs.append(-1)
                    if ti not in mlabel2:
                        c += 1
                        cs[ti] = conarea(subwin[mi], track_windows[ti])

                if c > 0:
                    css = np.argsort(cs)
                    tii = css[len(cs) - 1]
                    cnar = cs[css[len(css) - 1]]
                    if cnar > 0.4:
                            track_windows[tii] = subwin[mi]
                            mlabel2[mi] = tii
                    else:
                        track_windows.append(subwin[mi])
                        mlabel2[mi] = person_num
                        person_num += 1
                    #print ('mlabel2[' + str(mi) + '] to ' + str(tii))
                    #print ('conarea: ', cnar)

                else:
                    track_windows.append(subwin[mi])
                    mlabel2[mi] = person_num
                    person_num += 1
                    #print ('mlabel2[' + str(mi) + '] is ' + str(mlabel2[mi]))
            y, x, h, w = subwin[mi]

            cv2.rectangle(canvas, (y, x), (y + h, x + w), (255, 255, 255), 4)
            cv2.putText(canvas, str(mlabel2[mi]), (y + 5, x - 10), fontType, 1, (255, 255, 255), 4)



        people = people2
        mlabel = mlabel2
        mlabels.append(mlabel)
        #print (mlabel)
        #plt.figure()
        #plt.imshow(canvas[:,:,[2,1,0]])
        #fig = matplotlib.pyplot.gcf()
        #fig.set_size_inches(12, 12)
        #canvas.savefig(path2 + '/frame' + str(fnum2) + '.jpg')
        cv2.imwrite(path2 + '/frame' + str(fnum2) + '.jpg', canvas)

    end_time = time.time()
    print ("Finish!")
    print ("elasped time: " + str(end_time - start_time))
