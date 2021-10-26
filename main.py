import src.find_corners as fc
import src.calibrator as calib
import src.optimizer as optimizer


def main():
    square_size = 0.025
    width = 9
    height = 6

    obj_pts, img_pts, img_shapes, img_names = fc.find_corners(square_size,width,height)

    cb = calib.Calibrator(img_pts, obj_pts)

    cam_intrinsics, ext_list, k, img_pts, obj_pts = cb.init_Calib()
    print(cam_intrinsics)
    print(k)
    # print(ext_list)


    opt = optimizer.Optimizer(cam_intrinsics,ext_list,k,img_pts,obj_pts)

    opt_cam_intrinsics, opt_k, opt_ext_list = opt.refine_all()   

    print(opt_cam_intrinsics)
    print(opt_k)




if __name__ == '__main__':
    try:
        main()
    except:
        KeyboardInterrupt




