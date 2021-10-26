import glob
import numpy as np
import cv2 as cv


def find_corners(square_size, width, height):
    objp = np.zeros((height*width,2),np.float32)
    objp[:,:2] = np.mgrid[0:width, 0:height].T.reshape(-1,2)

    objp = objp * square_size



    # Array to store obj points and img points from all images
    obj_pts = []        # 3d pt in world
    img_pts = []        # 2d pt in img
    img_names = []
    img_shapes = []

    images = glob.glob('./img/Chessboard/*.jpg')

    cnt = 1

    criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER,30,0.001)


    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img_shapes.append([img.shape[0],img.shape[1]])
        img_names.append(fname)

        ret, corners = cv.findChessboardCorners(gray,(width,height),None)

        if ret:
            obj_pts.append(objp)
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            img_pts.append(corners2)

            # Draw and display the corners
            img = cv.drawChessboardCorners(img, (width,height), corners2, ret)
            
            cv.imwrite('./img/Detection/chessboard_' + str(cnt) + '.png',img)
        
            cnt+=1

    return obj_pts, img_pts, img_shapes, img_names

