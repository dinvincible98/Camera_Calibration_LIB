import numpy as np
from src.utils import *
from scipy import optimize



class Optimizer:
    def __init__(self, cam_intrinsics, ext_list, k, img_pts, obj_pts):
        self.cam_intrinsics = cam_intrinsics
        self.ext_list = ext_list
        self.k = k
        self.img_pts = img_pts
        self.obj_pts = obj_pts


    def refine_all(self):
        P_init = compose_parameter_vector(self.cam_intrinsics,self.k,self.ext_list)
        # print(P_init)
        X = np.zeros((2*len(self.obj_pts) * len(self.obj_pts[0]),2))
        Y_dot = np.zeros((2*len(self.obj_pts) * len(self.obj_pts[0])))
        
        M = len(self.obj_pts)           # num of views
        N = len(self.obj_pts[0])        # num of point in one view
        for i in range(M):
            for j in range(N):
                X[(i*N+j)*2] = self.obj_pts[i][j]
                X[(i*N+j)*2+1] = self.obj_pts[i][j]
                Y_dot[(i*N+j)*2] = self.img_pts[i][j][0][0]
                Y_dot[(i*N+j)*2+1] = self.img_pts[i][j][0][1]
        
        # print(X)
        # print(Y_dot)
        # LM optimize
        P = optimize.leastsq(refine_cost_func,
                             P_init,
                             args=(self.ext_list,self.img_pts,self.obj_pts),
                             Dfun=refine_jacobian_func)[0]
        # print(P)
        
        error = refine_cost_func(P,self.ext_list,self.img_pts,self.obj_pts)
        radial_error = [np.sqrt(error[2*i]*error[2*i] + error[2*i+1]*error[2*i+1]) for i in range(len(error)//2)]
        # print(radial_error)

        cam_intrinsics, k, ext_list = decompose_parameter_vector(P)

        return cam_intrinsics,k,ext_list



        
