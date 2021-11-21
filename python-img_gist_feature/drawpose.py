import numpy as np
import sys
import matplotlib.pyplot as plt

def load_poses(pose_path):
    """ Load ground truth poses (T_w_cam0) from file.
      Args:
        pose_path: (Complete) filename for the pose file
      Returns:
        A numpy array of size nx4x4 with n poses as 4x4 transformation
        matrices
  """
    # Read and parse the poses
    poses = []
    try:
        if '.txt' in pose_path:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)['arr_0']

    except FileNotFoundError:
        print('Ground truth poses are not avaialble.')

    return np.array(poses)


def load_calib(calib_path):
    """ Load calibrations (T_cam_velo) from file.
  """
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Tr:' in line:
                    line = line.replace('Tr:', '')
                    T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print('Calibrations are not avaialble.')

    return np.array(T_cam_velo)


def calculate_poses(poses_file, calib_file):
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    poses = load_poses(poses_file)
    pose_0 = poses[0]
    inv_pose_0 = np.linalg.inv(pose_0)
    pose_new = []

    for pose in poses:
        pose_new.append(T_velo_cam.dot(inv_pose_0).dot(pose).dot(T_cam_velo))
    pose_new = np.array(pose_new)
    return pose_new


rootpath = "/home/wy-lab/kitti_odometry_dataset/sequences/"
sequence = sys.argv[1]
poses_file = rootpath + sequence + "/" + sequence + ".txt"
calib_file = rootpath + sequence + "/calib.txt"
pose_new = calculate_poses(poses_file, calib_file)
offset = 5  # meters
map_size = [min(pose_new[:, 0, 3]) - offset,
            max(pose_new[:, 0, 3]) + offset,
            min(pose_new[:, 1, 3]) - offset,
            max(pose_new[:, 1, 3]) + offset]

fig, ax = plt.subplots()
plt.suptitle(sequence + " " + ' result', fontsize=15)
ax.set(xlim=map_size[:2], ylim=map_size[2:])
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.plot(pose_new[:, 0, 3], pose_new[:, 1, 3], '--', alpha=0.5, c='black', label='trajectory')
plt.show()