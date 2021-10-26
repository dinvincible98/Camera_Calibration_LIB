# Utils Functions
import numpy as np
import math

def get_transformation_matrix(pts, pt_type):
    x , y = 0, 0

    if pt_type == 0:
        x, y = pts[:,0][:,0], pts[:,0][:,1]
    else:
        x, y = pts[:,0], pts[:,1]

    
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    var_x = np.var(x)
    var_y = np.var(y)

    sx = np.sqrt(2.0 / var_x)
    sy = np.sqrt(2.0 / var_y)

    # Transformation matrix
    Nx = np.array([[sx, 0.0,-sx*mean_x],
                    [0.0, sy, -sy*mean_y],
                    [0.0, 0.0, 1.0]])

    return Nx


def homo_cost_func(coordinates, *params):
    h11, h12, h13, h21, h22, h23, h31, h32, h33 = params

    N = coordinates.shape[0] // 2
    
    X = coordinates[:N]
    Y = coordinates[N:]
    
    w = h31*X + h32*Y + h33
    x = (h11*X + h12*Y + h13) / w
    y = (h21*X + h22*Y + h23) / w

    res = np.zeros_like(coordinates)
    res[:N] = x
    res[N:] = y

    # print(res)
    return res


def homo_jacobian_func(coordinates, *params):
    h11, h12, h13, h21, h22, h23, h31, h32, h33 = params
    N = coordinates.shape[0] // 2
    
    X = coordinates[:N]
    Y = coordinates[N:]

    J = np.zeros((2*N,9))
    J_x = J[:N]
    J_y = J[N:]


    s_x = h11*X + h12*Y + h13
    s_y = h21*X + h22*Y + h23
    w = h31*X + h32*Y + h33

    J_x[:,0] = X / w
    J_x[:,1] = Y / w
    J_x[:,2] = 1 / w
    J_x[:,6] = -s_x*X / (w*w)
    J_x[:,7] = -s_x*Y / (w*w)
    J_x[:,8] = -s_x / (w*w)

    J_y[:,3] = X / w
    J_y[:,4] = Y / w
    J_y[:,5] = 1 / w 
    J_y[:,6] = -s_y*X / (w*w)
    J_y[:,7] = -s_y*Y / (w*w)
    J_y[:,8] = -s_y / (w*w)
    
    return J 


def create_v_ij(i,j,h_list):
    v_ij = np.zeros((h_list.shape[0],6))
    
    v_ij[:,0] = h_list[:,0,i] * h_list[:,0,j]
    v_ij[:,1] = h_list[:,0,i]*h_list[:,1,j] + h_list[:,1,i]* h_list[:,0,j]
    v_ij[:,2] = h_list[:,1,i] * h_list[:,1,j]
    v_ij[:,3] = h_list[:,2,i]*h_list[:,0,j] + h_list[:,0,i]*h_list[:,2,j]
    v_ij[:,4] = h_list[:,2,i]*h_list[:,1,j] + h_list[:,1,i]*h_list[:,2,j]
    v_ij[:,5] = h_list[:,2,i] * h_list[:,2,j]   

    return v_ij  


def to_homogenous_pts(pts):
    pts = np.atleast_2d(pts)

    N = pts.shape[0]
    pts_hom = np.hstack((pts,np.ones((N,1))))

    return pts_hom



def to_homogeneous_3d_pts(pts):
    if(pts.ndim !=2 or pts.shape[-1]!=2):
        raise ValueError("Must be 2d inhomogenous")
    
    N = pts.shape[0]
    pts_3d = np.hstack((pts,np.zeros((N,1))))
    # print(pts_3d)
    pts_3d_hom = to_homogenous_pts(pts_3d)
    return pts_3d_hom


def to_inhomogenous_pts(pts):
    pts = np.atleast_2d(pts)

    N = pts.shape[0]
    pts /= pts[:,-1][:,np.newaxis]
    pts_inhom = pts[:,:-1]

    return pts_inhom


def to_rodrigues_vec(rot_mat):
    p = 0.5 * np.array([[rot_mat[2][1]-rot_mat[1][2]],
                        [rot_mat[0][2]-rot_mat[2][0]],
                        [rot_mat[1][0]-rot_mat[0][1]]])
    c = 0.5 * (np.trace(rot_mat)-1)
    
    # print(p)
    # print(c)

    if np.linalg.norm(p) == 0:
        if c == 1:
            rot_vec = np.array([0,0,0])
        elif c == -1:
            rot_mat_plus = rot_mat + np.eye(3,3,dtype='float')
            norm_arr = np.array([np.linalg.norm(rot_mat_plus[:,0]), 
                                 np.linalg.norm(rot_mat_plus[:,1]),
                                 np.linalg.norm(rot_mat_plus[:,2])])
            v = rot_mat_plus[:, np.where(norm_arr==max(norm_arr))]
            u = v / np.linalg.norm(v)
            # print(u)
            u0, u1, u2 = u[0], u[1], u[2]

            if u0<0 or (u0==0 and u1<0) or (u0==0 and u1==0 and u2<0):
                u = -u
            rot_vec = math.pi * u
        else:
            rot_vec = []
    else:
        u = p / np.linalg.norm(p)
        # print(u)
        theta = math.atan2(np.linalg.norm(p),c)
        
        rot_vec = theta * u

    return rot_vec 

def to_rotation_matrix(rot_vec):
    theta = np.linalg.norm(rot_vec)
    rot_vec_hat = rot_vec / np.linalg.norm(rot_vec)         # unit vector
    rot_x, rot_y, rot_z = rot_vec_hat[0], rot_vec_hat[1], rot_vec_hat[2]
    W = np.array([[0, -rot_z, rot_y],
                 [rot_z, 0, -rot_z],
                 [-rot_y, rot_x, 0]])
    R = np.eye(3,dtype=np.float32) + W*math.sin(theta) + W*W*(1-math.cos(theta))

    return R


def compose_parameter_vector(cam_intrinsics, k, ext_list):
    a = np.array([cam_intrinsics[0][0], cam_intrinsics[1][1], cam_intrinsics[0][1], 
                  cam_intrinsics[0][2], cam_intrinsics[1][2], k[0], k[1]])
    
    P = a
    M = len(ext_list)
    for i in range(M):
        R, t = ext_list[i][:,:3], ext_list[i][:,3]
        # print(R)
        # print(t)
        rot_vec = to_rodrigues_vec(R)

        w = np.append(rot_vec,t)
        P = np.append(P,w)

    return P


def decompose_parameter_vector(P):
    cam_intrinsics = np.array([[P[0],P[2],P[3]],
                               [0, P[1], P[4]],
                               [0, 0, 1]])
    k = np.array([P[5], P[6]])
    W = []                          # list of R|t matrix
    M = (len(P) - 7) // 6           # num of extrinsics in list

    for i in range(M):
        m = 7 + 6*i
        rot_vec = P[m:m+3]
        t = np.reshape(P[m+3:m+6],(3,-1))
        R = to_rotation_matrix(rot_vec)
        R_t = np.concatenate((R,t),axis=1)
        W.append(R_t)
    
    return cam_intrinsics, k, W


def get_project_coordinates(cam_intrinsics, ext, k, coord):
    coor = np.array([coord[0],coord[1],0,1])

    coor_norm = np.dot(ext,coor)
    coor_norm /= coor_norm[-1]
    
    r = np.linalg.norm(coor_norm) 
    
    uv = np.dot(np.dot(cam_intrinsics,ext),coor)
    uv /= uv[-1]

    u0 = uv[0]
    v0 = uv[1]

    uc = cam_intrinsics[0][2]
    vc = cam_intrinsics[1][2]

    u = u0 + (u0-uc)*r*r*k[0] + (u0-uc)*r*r*r*r*k[1]
    v = v0 + (v0-vc)*r*r*k[0] + (v0-vc)*r*r*r*r*k[1]

    return np.array([u,v])




def refine_cost_func(P, W, img_pts, obj_pts):
    M = (len(P)-7) // 6         # num of views
    N = len(obj_pts[0])         # num of model pts
    cam_intrinsics = np.array([[P[0], P[2], P[3]],
                              [0, P[1], P[4]],
                              [0, 0, 1]])
    
    k = np.array(P[5:7])
    Y = np.array([])

    # print(k)

    for i in range(M):
        m = 7 + 6*i
        w = P[m:m+6]
        W_curr = W[i]
        for j in range(N):
            Y = np.append(Y,get_project_coordinates(cam_intrinsics,W_curr,k,obj_pts[i][j]))
    
    error_Y = np.array(img_pts).reshape(-1) - Y

    # print(error_Y)

    return error_Y



def refine_jacobian_func(P, W, img_pts, obj_pts):
    M = (len(P)-7) // 6         # num of views
    N = len(obj_pts[0])         # num of model pts
    K = len(P)
    cam_intrinsics = np.array([[P[0], P[2], P[3]],
                  [0, P[1], P[4]],
                  [0, 0, 1]])
    dist = np.array(P[5:7])
    # print(K)

    res = np.array([])
    
    for i in range(M):
        m = 7 + 6*i
        w = P[m:m+6]
        R = to_rotation_matrix(w[:3])
        # print(R)
        t = w[3:].reshape(3,1)
        # print(t)
        W_curr = np.concatenate((R,t),axis=1)
        # print(W_curr)

        for j in range(N):
            res = np.append(res, get_project_coordinates(cam_intrinsics,W_curr,dist,obj_pts[i][j]))
    # print(res)

    J = np.zeros((K, 2*M*N))
    for k in range(K):
        J[k] = np.gradient(res,P[k])

    # print(J)
    return np.transpose(J)    

            
