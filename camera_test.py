from matplotlib.animation import FuncAnimation, ArtistAnimation
from IPython.display import HTML
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import glob
import casadi as ca
from casadi import *
from scipy import *
import so3 as so3
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
cam = 0
cam_param = [642.0926, 642.0926, 1000.5, 1000.5,0]


# 321 Euler sequence
def euler2dcm(euler):
    phi = euler[0]
    theta = euler[1]
    psi = euler[2]

    dcm = np.array([
        [np.cos(theta) * np.cos(psi), np.cos(theta) * np.sin(psi), -np.sin(theta)],
        [-np.cos(phi) * np.sin(psi) + np.sin(phi) * np.sin(theta) * np.cos(psi),
        np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi), np.sin(phi) * np.cos(theta)],
        [np.sin(phi) * np.sin(psi) + np.cos(phi) * np.sin(theta) * np.cos(psi),
         -np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(theta) * np.sin(psi), np.cos(phi) * np.cos(theta)]
    ])
    return dcm


def cart2hom(point):
    return np.hstack([point, 1.0])


def hom2cart(coord):
    return coord[0:-1]


def get_cam_in(cam_param):
    fx = cam_param[0]
    fy = cam_param[1]
    cx = cam_param[2]
    cy = cam_param[3]

    s = cam_param[4]
    cam_in = np.array([
        [fx, s, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])
    return cam_in


def get_cam_ex(cam_att):
    rot_world2model = euler2dcm(cam_att)
    rot_model2cam = - np.array([
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
    ])

    cam_ex = rot_model2cam @ rot_world2model
    return cam_ex


def get_cam_mat(cam_param, cam_pose):
    assert (len(cam_param) == 5 and len(cam_pose) == 6)
    fx = cam_param[0]
    fy = cam_param[1]
    cx = cam_param[2]
    cy = cam_param[3]
    s = cam_param[4]

    pos = cam_pose[0:3]
    euler = cam_pose[3:6]

    # intrinsic matrix
    cam_in = np.array([
        [fx, s, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])

    # extrinsic matrix
    rot_world2model = euler2dcm(euler)
    rot_model2cam = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
    ])

    cam_ex = rot_model2cam @ rot_world2model
    cam_mat = cam_in @ np.block([cam_ex, -(cam_ex @ pos).reshape(-1, 1)])
    return cam_mat


def draw_box(img, coords):
    #     p_h = cam_mat@cart2hom(drone_pos)
    #     p_h = p_h/p_h[-1]

    #     p = hom2cart(p_h)
    box_size = 10
    color = (255,255,0)
    for row in coords:
        start = (row - [box_size, box_size]).astype(np.int32)
        end = (row + [box_size, box_size]).astype(np.int32)

        thickness = 1
        img = cv.rectangle(img, start, end, color, thickness)

    return img


def get_frame_dir():
    return 'data/frames/'


def get_cam_poses():
    return np.array([
        [20, 20, 12, 0, 0, -2.2],
        [20, -15, 12, 0, 0, 2.2],
        [-20, -20, 12, 0, 0, 0.7],
        [-20, 20, 12, 0, 0, -0.7],
    ])


def get_cam_dir():
    cam_dir = []
    for i in range(1,get_n_cam()+1):
        cam_dir.append(glob.glob(get_frame_dir() + 'camera' + str(i) + "/*.jpg"))
        cam_dir[i - 1].sort()
    return cam_dir


def get_n_cam():
    return 4


def get_n_frames():
    np.set_printoptions(precision=4)
    # [x y z roll pitch yaw]
    cam_mat = get_cam_mat(cam_param, get_cam_poses()[cam])
    # n_frames = min([len(cam_dir[i]) for i in range(n_cam)])
    return len(get_cam_dir()[cam])


def get_drone_coords(n_frames):
    drone_coords = []
    for i in range(n_frames):
        elapsed = i*0.1
        x = 30*np.sin(2*np.pi*elapsed*1/3)
        y = 15*np.cos(2*np.pi*elapsed*1/3)
        z = 20
        drone_coords.append([x,y,z])
    return np.array(drone_coords)


def animation(pixels1):
    fig, ax = plt.subplots(figsize=(15,15))

    ims = []
    cam_imgs = get_cam_dir()[cam]
    for i in range(get_n_frames()):
        plt.figure(figsize=(15,15))
        img = plt.imread(cam_imgs[i])
        img_detect = draw_box(img, pixels1)
        im = ax.imshow(img_detect)
        ims.append([im])

    ani = ArtistAnimation(fig, ims, interval=100)
    HTML(ani.to_html5_video())


def get_pixel_vals(drone_coords, n_frames):
    cam_mat = get_cam_mat(cam_param, get_cam_poses()[cam])
    drone_coords_hom = np.hstack([drone_coords, np.ones((n_frames,1))])
    pixel_coords_hom = (cam_mat @ drone_coords_hom.T).T
    pixel_coords = []
    for coord in pixel_coords_hom:
        pixel_coords.append([coord[0]/coord[2], coord[1]/coord[2]])
    return np.array(pixel_coords)


def plot(pixel_coords):
    fig = plt.figure()
    plt.plot(pixel_coords[:,0], pixel_coords[:,1], '.')
    plt.axis('square')
    plt.xlim(0,2000)
    plt.ylim(2000,0)
    plt.show()


def detect(pixel_coords):
    cam_imgs = get_cam_dir()[cam]
    plt.figure(figsize=(15,15))
    img = plt.imread(cam_imgs[1])
    img_detect = draw_box(img, pixel_coords)
    plt.imshow(img_detect)


def euler2dcm_ca(euler):
    from casadi import sin,cos
    phi = euler[0]
    phi = euler[0]
    theta = euler[1]
    psi = euler[2]

    dcm = ca.SX(3, 3)
    dcm[0, 0] = cos(theta) * cos(psi)
    dcm[0, 1] = cos(theta) * sin(psi)
    dcm[0, 2] = -sin(theta)
    dcm[1, 0] = -cos(phi) * sin(psi) + sin(phi) * sin(theta) * cos(psi)
    dcm[1, 1] = cos(phi) * cos(psi) + sin(phi) * sin(theta) * sin(psi)
    dcm[1, 2] = sin(phi) * cos(theta)
    dcm[2, 0] = sin(phi) * sin(psi) + cos(phi) * sin(theta) * cos(psi)
    dcm[2, 1] = -sin(phi) * cos(psi) + cos(phi) * sin(theta) * sin(psi)
    dcm[2, 2] = cos(phi) * cos(theta)
#     dcm = ca.MX([
#         [cos(theta)*cos(psi), cos(theta)*sin(psi), -sin(theta)],
#         [-cos(phi)*sin(psi)+sin(phi)*sin(theta)*cos(psi), cos(phi)*cos(psi)+s>
#         [sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi), -sin(phi)*cos(psi)+c>
    #     ])
    return dcm


def get_cam_mat_ca(cam_param, cam_pos, cam_att):
    #     assert(len(cam_param)==5 and len(cam_pose)==6)
    fx = cam_param[0]
    fy = cam_param[1]
    cx = cam_param[2]
    cy = cam_param[3]
    s = cam_param[4]

    pos = cam_pos
    euler = cam_att

    # intrinsic matrix
    cam_in = ca.SX(3, 3)
    cam_in[0, 0] = fx
    cam_in[0, 1] = s
    cam_in[0, 2] = cx
    cam_in[1, 1] = fy
    cam_in[1, 2] = cy
    cam_in[2, 2] = 1
    # extrinsic matrix
    rot_world2model = euler2dcm_ca(euler)
    rot_model2cam = ca.SX(np.array([
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
    ]))

    cam_ex = rot_model2cam @ rot_world2model
    block = ca.SX(3, 4)
    block[:, 0:3] = cam_ex
    block[:, 3] = -(cam_ex @ pos)
    cam_mat = cam_in @ block
    return cam_mat


def get_cam_mat_lie_ca(cam_param, cam_pos, cam_lie):
    #     assert(len(cam_param)==5 and len(cam_pose)==6)
    fx = cam_param[0]
    fy = cam_param[1]
    cx = cam_param[2]
    cy = cam_param[3]
    s = cam_param[4]

    pos = cam_pos
    w = cam_lie  # so3

    # intrinsic matrix
    cam_in = ca.SX(3, 3)
    cam_in[0, 0] = fx
    cam_in[0, 1] = s
    cam_in[0, 2] = cx
    cam_in[1, 1] = fy
    cam_in[1, 2] = cy

    cam_in[2, 2] = 1
    # extrinsic matrix
    #     rot_world2model = euler2dcm_ca(euler)
    rot_world2model = so3.Dcm.exp(w)
    rot_model2cam = ca.SX(np.array([
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
    ]))

    cam_ex = rot_model2cam @ rot_world2model
    block = ca.SX(3, 4)
    block[:, 0:3] = cam_ex
    block[:, 3] = -(cam_ex @ pos)
    cam_mat = cam_in @ block
    return cam_mat


def run_euler():
    w = ca.SX.sym('w', 3,1)
    dcm = ca.SX.sym('dcm', 3,3)
    euler = ca.SX.sym('euler', 3,1)
    f_lie2dcm = ca.Function('f_lie2dcm', [w], [so3.Dcm.exp(w)])
    f_dcm2lie = ca.Function('f_dcm2lie', [dcm], [so3.Dcm.log(dcm)])
    f_euler2dcm = ca.Function('f_euler2dcm', [euler], [so3.Dcm.from_euler(euler)])
    f_dcm2euler = ca.Function('f_dcm2euler', [dcm], [so3.Euler.from_dcm(dcm)])
    f_lie2euler = ca.Function('f_lie2euler', [w], [so3.Euler.from_dcm(so3.Dcm.exp(w))])
    f_euler2lie = ca.Function('f_euler2lie', [euler], [so3.Dcm.log(so3.Dcm.from_euler(euler))])
    print(f_euler2lie(cam_poses[3][3:6]))

    cam_param_ca = ca.SX.sym('param', 5)
    cam_pos_ca = ca.SX.sym('pos', 3)
    cam_att_ca = ca.SX.sym('att', 3)
    cam_lie_ca = ca.SX.sym('lie', 3)

    cam_mat_ca = get_cam_mat_ca(cam_param_ca, cam_pos_ca, cam_att_ca)
    cam_mat_lie_ca = get_cam_mat_lie_ca(cam_param_ca, cam_pos_ca, cam_lie_ca)
    f_cam_mat = ca.Function('f_cam_mat',[cam_param_ca, cam_pos_ca, cam_att_ca], [cam_mat_ca])
    f_cam_mat_lie = ca.Function('f_cam_mat_lie',[cam_param_ca, cam_pos_ca, cam_lie_ca], [cam_mat_lie_ca])
    f_cam_mat(cam_param, cam_poses[3][0:3], cam_poses[3][3:6])

    cam_mat = get_cam_mat(cam_param, cam_poses[3])
    p1_hom = cam_mat @ cart2hom(cam_poses[1][0:3])
    p1 = p1_hom[0:2]/p1_hom[2]
    p2_hom = cam_mat @ cart2hom(cam_poses[2][0:3])
    p2 = p2_hom[0:2]/p2_hom[2]

    cam_mat_ca = get_cam_mat_ca(cam_param, cam_poses[3][0:3], cam_att_ca) 
    p1_hom_ca = cam_mat_ca @ cart2hom(cam_poses[1][0:3])
    p2_hom_ca = cam_mat_ca @ cart2hom(cam_poses[2][0:3])
    p1_ca = p1_hom_ca[0:2]/p1_hom_ca[2]
    p2_ca = p2_hom_ca[0:2]/p2_hom_ca[2]

    # optimization over euler angles
    nlp = {'x':cam_att_ca, 'f': ca.norm_2(ca.vertcat(p1-p1_ca, p2-p2_ca))**2, 'g':0}
    S = ca.nlpsol('S', 'ipopt', nlp, {
        'print_time': 0,
            'ipopt': {
                'sb': 'yes',
                'print_level': 0,
                }
    })

    r = S(x0=[0,0,-0.1], lbg=0, ubg=0)
    x_opt = r['x']
    print('x_opt: ', x_opt)
    print(cam_poses[3][3:6])

    # optimization over so3
    cam_mat_lie_ca = get_cam_mat_lie_ca(cam_param, cam_poses[3][0:3], cam_lie_ca)

    p1_hom_ca = cam_mat_lie_ca @ cart2hom(cam_poses[1][0:3])
    p2_hom_ca = cam_mat_lie_ca @ cart2hom(cam_poses[2][0:3])
    p1_ca = p1_hom_ca[0:2]/p1_hom_ca[2]
    p2_ca = p2_hom_ca[0:2]/p2_hom_ca[2]

    nlp = {'x':cam_lie_ca, 'f': ca.norm_2(ca.vertcat(p1-p1_ca, p2-p2_ca))**2, 'g':0}
    S = ca.nlpsol('S', 'ipopt', nlp, {
        'print_time': 0,
            'ipopt': {
                'sb': 'yes',
                'print_level': 0,
                }
    })

    r = S(x0=[0,0, -0.1], lbg=0, ubg=0)
    x_opt = r['x']
    print('x_opt: ', x_opt)
    print(cam_poses[3][3:6])
    np.rad2deg(-0.46)


def get_vec_true():
    return drone_coords1[0,:] - get_cam_poses()[cam][0:3]


def get_dist_true(vec_true):
    return np.linalg.norm(vec_true)

def ray_casting(drone_coords):
    K = get_cam_in(cam_param)
    R = get_cam_ex(get_cam_poses()[cam][3:6])
    Bp = np.linalg.inv(K @ R)

    vec = Bp @ cart2hom(pixels[0,:]).astype('int32') # cast to integer because actual pixels are discrete
    vec = vec/np.linalg.norm(vec)
    print(vec)

    vec_true = get_vec_true()
    dist_true = get_dist_true(vec_true)
    vec_true = vec_true/np.linalg.norm(vec_true)
    print(vec_true)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot3D(drone_coords[:,0], drone_coords[:,1], drone_coords[:,2],'.')
    ax.plot3D(get_cam_poses()[:,0], get_cam_poses()[:,1], get_cam_poses()[:,2], 'rx')
    ax.quiver(get_cam_poses()[cam,:][0], get_cam_poses()[cam,:][1], get_cam_poses()[cam,:][2], vec[0], vec[1], vec[2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

def range_finding():
    print(get_dist_true(get_vec_true()))
    # suppose the bounding box is 40x40 pixels
    dimension = np.array([550, 550, 110])
    box_size = np.array([40,40])


n_frames1 = get_n_frames()
drone_coords1 = get_drone_coords(n_frames1)
pixels = get_pixel_vals(drone_coords1, n_frames1)
print(pixels)
plot(pixels)
# animation(pixels)
ray_casting(drone_coords1)
range_finding()
