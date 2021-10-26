import numpy as np
import cv2 as cv
from numpy.lib.type_check import imag
from scipy.optimize.minpack import curve_fit
from src.utils import *

class Calibrator:
    def __init__(self,img_pts,obj_pts):
        self.img_pts = img_pts
        self.obj_pts = obj_pts
        self.refined_homo_list = []
        self.extrinsics_list = []

    def init_Calib(self):
        refined_homo_list = self.get_Homography()
        # print(refined_homo_list)
        
        cam_intrinsics = self.cal_Camera_Intrinsics(refined_homo_list)

        # print(cam_intrinsics)

        R_t_list = self.cal_Extrinsics(cam_intrinsics,refined_homo_list)
        # print(R_t_list)

        k = self.estimate_Distortion(cam_intrinsics,R_t_list)
        # print(k)

        return cam_intrinsics,R_t_list,k,self.img_pts,self.obj_pts
    
    def get_Homography(self):
        for i in range(len(self.img_pts)):
            h_mat = self.estimate_Homography(self.img_pts[i],self.obj_pts[i])
            # print(h_mat)
            refined_h_mat = self.refine_Homography(h_mat,self.img_pts[i],self.obj_pts[i])
            # print(refined_h_mat)

            self.refined_homo_list.append(refined_h_mat)

        return self.refined_homo_list


    def estimate_Homography(self, img_pts, obj_pts):
        
        N = len(img_pts)

        # Get normalization matrix
        norm_obj = get_transformation_matrix(obj_pts,1)
        norm_img = get_transformation_matrix(img_pts,0)


        a_list = []
        for i in range(N):
            # Get homogenous matrix
            homo_obj = np.matrix([obj_pts[i][0],obj_pts[i][1],1])
            homo_img = np.matrix([img_pts[i][0][0],img_pts[i][0][1],1])


            # Normalize 
            obj_prime = np.dot(homo_obj,np.transpose(norm_obj))
            img_prime = np.dot(homo_img,np.transpose(norm_img))

            # print(img_prime)
            # print(obj_prime)

            model_x, model_y = obj_prime.item(0), obj_prime.item(1)
            img_x, img_y = img_prime.item(0), img_prime.item(1)

            # Form 2N x 9 matrix by stacking rows
            a_row1 = [model_x, model_y, 1, 0, 0, 0, -model_x*img_x, -model_y*img_x, -img_x] 
            a_row2 = [0, 0, 0, model_x, model_y, 1, -model_x*img_y, -model_y*img_y, -img_y]

            # print(a_row1)
            # print(a_row2)

            a_list.append(a_row1)
            a_list.append(a_row2)


        # print(a_list)
        # Convert list to matrix
        A_mat = np.matrix(a_list)

        # Solve svd
        U, s, V = np.linalg.svd(A_mat)

        # Find the smallest line
        idx = np.argmin(s)
        L = V[idx]

        H_mat = np.reshape(L,(3,3))     # 3x3 homography matrix

        # De-normalize
        H_mat = np.dot(np.dot(np.linalg.inv(norm_img),H_mat),norm_obj)

        return H_mat    


    def refine_Homography(self, h_mat, img_pts, obj_pts):
        u, v = img_pts[:,0][:,0], img_pts[:,0][:,1]       # img_pts
        X, Y = obj_pts[:,0], obj_pts[:,1]                 # obj pts

        N = X.shape[0]
        
        h0 = np.ravel(h_mat)             # flatten array

        x_points = np.zeros(2*N)
        x_points[:N] = X
        x_points[N:] = Y

        y_points = np.zeros(2*N)
        y_points[:N] = u
        y_points[N:] = v

        # scipy LM optimizer
        popt, pcov = curve_fit(homo_cost_func,
                               x_points,
                               y_points,
                               p0=h0,
                               jac=homo_jacobian_func,
                               method='lm',
                               maxfev=5000)

        h_refined = popt

        # Normalize 
        h_refined /= h_refined[-1]

        # Reshape
        h_refined = np.reshape(h_refined,(3,3))

        return h_refined


    def cal_Camera_Intrinsics(self, h_list):
        M = len(h_list)
        
        homo_list = np.zeros((M,3,3))
        
        for idx, h in enumerate(h_list):
            homo_list[idx] = h
        
        # Generate homogenous equations
        v00 = create_v_ij(0,0,homo_list)
        v01 = create_v_ij(0,1,homo_list)
        v11 = create_v_ij(1,1,homo_list)


        V = np.zeros((2*M,6))
        V[:M] = v01
        V[M:] = v00 - v11

        # Solve svd
        U, s, V = np.linalg.svd(V)
        idx = np.argmin(s)

        b0, b1, b2, b3, b4, b5 = V[idx]

        # w = b0*b2*b5 - b1*b1*b5 - b0*b4*b4 + 2*b1*b3*b4 - b2*b3*b3
        # d = b0*b2 - b1*b1
        # alpha = np.sqrt(w/(d*b0))
        # beta = np.sqrt(w/(d*d)*b0)
        # gamma = np.sqrt(w/(d*d*b0)) * b1
        # uc = (b1*b4 - b2*b3) / d
        # vc = (b1*b3 - b0*b4) / d

        v0 = (b1 * b3 - b0 * b4) / (b0 * b2 - b1 * b1)
        lmbda = b5 - (b3 * b3 + v0 * (b1 * b3 - b0 * b4)) / b0
        alpha = np.sqrt(lmbda / b0)
        beta = np.sqrt(lmbda * b0 / (b0 * b2 - b1 * b1))
        gamma = -b1 * alpha * alpha * beta / lmbda
        u0 = gamma * v0 / beta - b3 * alpha * alpha / lmbda

        A = np.eye(3,3,dtype=np.float32)
        A[0][0] = alpha
        A[0][1] = gamma
        A[0][2] = u0
        A[1][1] = beta
        A[1][2] = v0

        return A 


    def cal_Extrinsics(self,cam_intrinsics, h_list):
        for h_mat in h_list:
            h0 = [row[0] for row in h_mat]
            h1 = [row[1] for row in h_mat]
            h2 = [row[2] for row in h_mat]


            A_inv = np.linalg.inv(cam_intrinsics)
            # print(A_inv)

            k = 1.0 / np.linalg.norm(np.dot(A_inv,h0))          # scalar
            r0 = k * np.dot(A_inv,h0)
            r1 = k * np.dot(A_inv,h1)
            r2 = np.cross(r0,r1)
            
            t = k * np.dot(A_inv,h2)
            R = np.vstack((r0,r1,r2)).T
            # print(R)

            # Solve svd
            U, s, V = np.linalg.svd(R)
            new_R = np.dot(U, V)

            # Form full extrinsics
            R_t = np.hstack((new_R,t[:,np.newaxis]))

            # print(R_t)

            self.extrinsics_list.append(R_t)

        return self.extrinsics_list


    def estimate_Distortion(self, cam_intrinsics, extrinsics_list):
        M = len(self.img_pts)
        N = self.obj_pts[0].shape[0]
        # print(N)

        # Convert to homogenous matrix
        model = to_homogeneous_3d_pts(self.obj_pts[0])

        uc, vc = cam_intrinsics[0][2], cam_intrinsics[1][2]

        # print(model)

        # Form radius vector
        r = np.zeros(2*M*N)
        for idx, ext in enumerate(extrinsics_list):
            norm_proj = np.dot(model, ext.T)
            norm_proj = to_inhomogenous_pts(norm_proj)

            x_norm_proj, y_norm_proj = norm_proj[:,0], norm_proj[:,1] 

            r_idx = np.sqrt(x_norm_proj*x_norm_proj + y_norm_proj*y_norm_proj)
            r[idx*N:(idx+1)*N] = r_idx
        
        r[M*N:] = r[:M*N]       # replicate value

        # print(r)
        # Form observation vector
        obs = np.zeros(2*M*N)
        # print(obs)
        u_data = np.zeros(M*N)
        v_data = np.zeros(M*N) 
        # print(u_data)
        # print(v_data)

        for idx, data in enumerate(self.img_pts):
            u_i, v_i = data[:,0][:,0], data[:,0][:,1]
            # print(u_i)
            u_data[idx*N:(idx+1)*N] = u_i
            v_data[idx*N:(idx+1)*N] = v_i
        obs[:M*N] = u_data
        obs[M*N:] = v_data
        # print(obs)
        # Form prediction vector
        pred = np.zeros(2*M*N)
        pred_centered = np.zeros(2*M*N)

        u_pred, v_pred = np.zeros(M*N), np.zeros(M*N)

        for idx,ext in enumerate(extrinsics_list):
            P = np.dot(cam_intrinsics,ext)          # projection matrix
            projection = np.dot(model,P.T)
            projection = to_inhomogenous_pts(projection)
            u_pred_i = projection[:,0]
            v_pred_i = projection[:,1]

            u_pred[idx*N:(idx+1)*N] = u_pred_i
            v_pred[idx*N:(idx+1)*N] = v_pred_i
        
        pred[:M*N] = u_pred
        pred[M*N:] = v_pred

        pred_centered[:M*N] = u_pred - uc
        pred_centered[M*N:] = v_pred - vc

        
        # Form distortion coefficient
        D = np.zeros((2*M*N,2))
        D[:,0] = pred_centered * r*r
        D[:,1] = pred_centered * r*r*r*r

        # Form values b (different between sensor observation and predictions)
        b = obs - pred

        # Use pseudo-inverse to compute least squares solutions for dist coefficients
        D_inv = np.linalg.pinv(D)
        k = np.dot(D_inv,b)

        return k




         


        












        

         





    









        




        