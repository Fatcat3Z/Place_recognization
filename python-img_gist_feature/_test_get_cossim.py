import os
import time
from scipy.spatial.distance import cdist
import cv2
import sys
import json
import numpy as np
import datetime
from tqdm import tqdm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
sys.path.append("./img_gist_feature/")

from utils_gist import *
from util__base import *
from util__cal import *


def get_img_gist_feat(s_img_url, colormode):
    gist_helper = GistUtils()
    # np_img = cv2.imread(s_img_url, -1)
    np_gist = gist_helper.get_gist_vec(s_img_url, mode=colormode)
    np_gist_L2Norm = np_l2norm(np_gist)
    # print()
    # print("img url: %s" % s_img_url)
    # print("shape ", np_gist.shape)
    # print("gist feature noly show 10dim", np_gist[0,:10], "...")
    # print("gist feature(L2 norm) noly show 10dim", np_gist_L2Norm[0,:10], "...")
    # print()
    return np_gist_L2Norm


def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))


def proc_main(O_IN):
    s_img_url_a = O_IN["s_img_url_a"]
    s_img_url_b = O_IN["s_img_url_b"]

    np_img_gist_a = get_img_gist_feat(s_img_url_a)
    np_img_gist_b = get_img_gist_feat(s_img_url_b)

    # f_img_sim = np.inner(np_img_gist_a, np_img_gist_b)
    # f_img_sim = cross_entropy(np_img_gist_a, np_img_gist_b)
    f_img_sim = cdist(np_img_gist_a, np_img_gist_b, metric='euclidean')
    print("%.23f" % f_img_sim)

    np_img_group = cv2.imread(s_img_url_a)
    np_img_public = cv2.imread(s_img_url_b)

    fig = plt.figure()
    plt.suptitle("%.7f" % f_img_sim, fontsize=10)

    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(np_img_group[:, :, ::-1])
    ax.set_title("%s" % s_img_url_a, fontsize=10)

    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(np_img_public[:, :, ::-1])
    ax.set_title("%s" % s_img_url_b, fontsize=10)
    fig.savefig("./test/show.png")
    plt.show()


def getimages(filepath):
    imgs = []
    files = []
    for file in os.listdir(filepath):
        files.append(file)
    files.sort()
    for file in files:
        img = cv2.imread(filepath + "/" + file)
        imgs.append(img)
    return imgs


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


def findpossiblematch(imgs, idx, range_vec, depthorder_vec, spatial_vec, normal_vec, scancontext_vec, mode):
    min_sim = 1
    min_index = 0
    if mode == 0:
        for i in range(idx):
            if idx - i < 100:
                continue
            scancontext_img_sim = cdist(imgs[2], scancontext_vec[i], metric='euclidean')
            # range_img_sim = cdist(imgs[0], range_vec[i], metric='euclidean')
            average_sim = scancontext_img_sim
            if average_sim < min_sim:
                min_sim = average_sim
                min_index = i
            # print("round" + str(i) + "：" + str(min_sim))
            if min_sim < 0.2:
                return min_sim, min_index
    elif mode == 1:
        for i in range(idx):
            if idx - i < 100:
                continue
            # range_img_sim = cdist(imgs[0], range_vec[i], metric='euclidean')
            normal_img_sim = cdist(imgs[1], normal_vec[i], metric='euclidean')
            scancontext_img_sim = cdist(imgs[2], scancontext_vec[i], metric='euclidean')
            # spatial_img_sim = cdist(imgs[2], spatial_vec[i], metric='euclidean')
            average_sim = (normal_img_sim + scancontext_img_sim) / 2
            if average_sim < min_sim:
                min_sim = average_sim
                min_index = i
            # print("round" + str(i) + "：" + str(min_sim))
            if min_sim < 0.2:
                return min_sim, min_index
    elif mode == 2:
        for i in range(idx):
            if idx - i < 100:
                continue
            range_img_sim = cdist(imgs[0], range_vec[i], metric='euclidean')
            normal_img_sim = cdist(imgs[1], normal_vec[i], metric='euclidean')
            # spatial_img_sim = cdist(imgs[2], spatial_vec[i], metric='euclidean')
            scancontext_img_sim = cdist(imgs[2], scancontext_vec[i], metric='euclidean')
            # average_sim = (range_img_sim + normal_img_sim + scancontext_img_sim) / 3
            average_sim = (range_img_sim + normal_img_sim + scancontext_img_sim) / 3
            if average_sim < min_sim:
                min_sim = average_sim
                min_index = i
            if min_sim < 0.2:
                return min_sim, min_index
    elif mode == 3:
        for i in range(len(range_vec)):
            if i == idx:
                continue
            range_img_sim = cdist(imgs[0], range_vec[i], metric='euclidean')
            normal_img_sim = cdist(imgs[3], normal_vec[i], metric='euclidean')
            spatial_img_sim = cdist(imgs[2], spatial_vec[i], metric='euclidean')
            scancontext_img_sim = cdist(imgs[4], scancontext_vec[i], metric='euclidean')
            average_sim = (range_img_sim +  normal_img_sim + scancontext_img_sim + spatial_img_sim) / 4
            if average_sim < min_sim:
                min_sim = average_sim
                min_index = i
            if min_sim < 0.1:
                return min_sim, min_index
    elif mode == 4:
        for i in range(len(range_vec)):
            if i == idx:
                continue
            range_img_sim = cdist(imgs[0], range_vec[i], metric='euclidean')
            # depthorder_img_sim = cdist(imgs[1], depthorder_vec[i], metric='euclidean')
            spatial_img_sim = cdist(imgs[2], spatial_vec[i], metric='euclidean')
            normal_img_sim = cdist(imgs[3], normal_vec[i], metric='euclidean')
            scancontext_img_sim = cdist(imgs[4], scancontext_vec[i], metric='euclidean')
            average_sim = (
                                  range_img_sim + spatial_img_sim + normal_img_sim + scancontext_img_sim) / 4
            if average_sim < min_sim:
                min_sim = average_sim
                min_index = i
            if min_sim < 0.1:
                return min_sim, min_index
    return min_sim, min_index


def matchwithyaw(rangemats, scancontextmats, checkidx, mode, rangevecs, scanvecs):
    min_sim = 1
    min_index = 0
    width = len(rangemats[0][0])
    if mode == 0:
        for i in range(checkidx):
            if checkidx - i < 30:
                continue
            _, yaw_unit = getinitialyaw(scancontextmats[i], scancontextmats[checkidx])
            range_new = np.hstack((rangemats[i][:, (width - yaw_unit):], rangemats[i][:, :(width - yaw_unit)]))
            if yaw_unit == 0:
                rangefeature = rangevecs[i]
            else:
                rangefeature = get_img_gist_feat(range_new, "gray")

            range_img_sim = cdist(rangefeature, range_vec[i], metric='euclidean')
            average_sim = range_img_sim
            if average_sim < min_sim:
                min_sim = average_sim
                min_index = i
            # print("round" + str(i) + "：" + str(min_sim))
            if min_sim < 0.1:
                return min_sim, min_index
    elif mode == 1:
        for i in range(checkidx):
            if checkidx - i < 35:
                continue
            _, yaw_unit = getinitialyaw(scancontextmats[i], scancontextmats[checkidx])
            scanew_new = np.hstack((scancontextmats[i][:, (width - yaw_unit):], scancontextmats[i][:, :(width - yaw_unit)]))
            if yaw_unit == 0:
                scanfeature = scanvecs[i]
            else:
                scanfeature = get_img_gist_feat(scanew_new, "gray")

            scancontext_img_sim = cdist(scanfeature, scancontext_vec[i], metric='euclidean')
            average_sim = (scancontext_img_sim)
            if average_sim < min_sim:
                min_sim = average_sim
                min_index = i
            # print("round" + str(i) + "：" + str(min_sim))
            if min_sim < 0.1:
                return min_sim, min_index
    elif mode == 2:
        for i in range(checkidx):
            if checkidx - i < 35:
                continue
            _, yaw_unit = getinitialyaw(scancontextmats[i], scancontextmats[checkidx])
            scanew_new = np.hstack((scancontextmats[i][:, (width - yaw_unit):], scancontextmats[i][:, :(width - yaw_unit)]))
            range_new = np.hstack((rangemats[i][:, (width - yaw_unit):], rangemats[i][:, :(width - yaw_unit)]))

            if yaw_unit == 0:
                scanfeature = scanvecs[i]
                rangefeature = rangevecs[i]
            else:
                scanfeature = get_img_gist_feat(scanew_new, "gray")
                rangefeature = get_img_gist_feat(range_new, "gray")

            range_img_sim = cdist(rangefeature, range_vec[i], metric='euclidean')
            scancontext_img_sim = cdist(scanfeature, scancontext_vec[i], metric='euclidean')
            average_sim = (range_img_sim + scancontext_img_sim) / 2
            if average_sim < min_sim:
                min_sim = average_sim
                min_index = i
            if min_sim < 0.1:
                return min_sim, min_index
    return min_sim, min_index

def getinitialyaw(checkscan, targetscan):
    width = len(targetscan[0])
    checkvec = []
    targetvec = []
    for i in range(len(targetscan[0])):
        checkvec.append(np.sum(checkscan[:, i]))
        targetvec.append(np.sum(targetscan[:, i]))
    checkvec = np.asarray(checkvec).reshape(1, -1)
    targetvec = np.asarray(targetvec).reshape(1, -1)
    check_norm = np_l2norm(checkvec)
    target_norm = np_l2norm(targetvec)
    # print(checkvec.shape)
    min_sim = cdist(check_norm, target_norm, metric='euclidean')
    yawunit = 0
    target_norm = np.hstack((target_norm[:, (width - 1):], target_norm[:, :(width - 1)]))

    for i in range(width-1):
        target_norm = np.hstack((target_norm[:, (width - 1):], target_norm[:, :(width - 1)]))
        sim = cdist(check_norm, target_norm, metric='euclidean')
        if sim < min_sim:
            min_sim = sim
            yawunit = i
    return min_sim, yawunit


def gethogdiscriptor(img):
    cell_size = (8, 8)  # h x w in pixels
    block_size = (2, 2)  # h x w in cells
    nbins = 9  # number of orientation bins

    # winSize is the size of the image cropped to an multiple of the cell size
    # _winSize 896 * 64
    # _blockSize 16 * 16
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
    hog_feats = hog.compute(img) \
        .reshape(n_cells[1] - block_size[1] + 1,
                 n_cells[0] - block_size[0] + 1,
                 block_size[0], block_size[1], nbins) \
        .transpose((1, 0, 2, 3, 4))  # index blocks by rows first
    # hog_feats now contains the gradient amplitudes for each direction,
    # for each cell of its group for each group. Indexing is by rows then columns.
    # 8 * 112 * 9
    gradients = np.zeros((n_cells[0], n_cells[1], nbins))

    # count cells (border cells appear less often across overlapping groups)
    cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

    for off_y in range(block_size[0]):
        for off_x in range(block_size[1]):
            gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
            off_x:n_cells[1] - block_size[1] + off_x + 1] += \
                hog_feats[:, :, off_y, off_x, :]
            cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
            off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

    # Average gradients
    gradients /= cell_count

    # Preview
    # plt.figure()
    # plt.imshow(img, cmap='gray')
    # plt.show()

    bin = 1  # angle is 360 / nbins * direction
    plt.pcolor(gradients[:, :, bin])
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.show()
    return gradients


def calculatehogloss(checkhogdes, matchhogdes):
    diffres = 0
    for i in range(9):
        diff = checkhogdes[:, :, i] - matchhogdes[:, :, i]
        diffres += sum(diff)
    return diffres

if __name__ == "__main__":
    rangefile = "/home/wy-lab/kitti_odometry_dataset/sequences/07/RmvImgs/RmvRangeMat"
    depthorderfile = "/home/wy-lab/kitti_odometry_dataset/sequences/07/RmvImgs/RmvDepthOrderMat"
    spatialsavepath = "/home/wy-lab/kitti_odometry_dataset/sequences/07/RmvImgs/RmvSpatialAreaMat"
    scancontextsavepath = "/home/wy-lab/kitti_odometry_dataset/sequences/07/RmvImgs/RmvScanContextMat"
    normalsavepath = "/home/wy-lab/kitti_odometry_dataset/sequences/07/RmvImgs/RmvNormalMat"
    poses_file = "/home/wy-lab/kitti_odometry_dataset/sequences/07/07.txt"
    calib_file = "/home/wy-lab/kitti_odometry_dataset/sequences/07/calib.txt"

    rangemat = getimages(rangefile)
    # depthordermat = getimages(depthorderfile)
    # spatialmat = getimages(spatialsavepath)
    scancontextmat = getimages(scancontextsavepath)
    # normalmat = getimages(normalsavepath)
    range_vec = []
    depthorder_vec = []
    spatial_vec = []
    normal_vec = []
    scancontext_vec = []
    # get gt pose files
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

    for i in range(len(rangemat)):
        range_vec.append(get_img_gist_feat(rangemat[i], "gray"))
        # depthorder_vec.append(get_img_gist_feat(depthordermat[i], "gray"))
        # spatial_vec.append(get_img_gist_feat(spatialmat[i], "gray"))
        # normal_vec.append(get_img_gist_feat(normalmat[i], "rgb"))
        scancontext_vec.append(get_img_gist_feat(scancontextmat[i], "gray"))
    #
    similarity_threshold = 0.15
    num_true_positive = 0
    num_false_positive = 0
    num_true_negative = 0
    num_false_negative = 0
    average_sim = 0
    runtime = 0
    tp_x = []
    tp_y = []
    fp_x = []
    fp_y = []
    tn_x = []
    tn_y = []
    fn_x = []
    fn_y = []

    for i in tqdm(range(len(rangemat))):
        checkplace = str(i).rjust(4, "0")
        checkplaceimg = []
        start = time.time()
        checkplaceimg.append(range_vec[int(checkplace)])
        # checkplaceimg.append(depthorder_vec[int(checkplace)])
        # checkplaceimg.append(spatial_vec[int(checkplace)])
        # checkplaceimg.append(normal_vec[int(checkplace)])
        checkplaceimg.append(scancontext_vec[int(checkplace)])
        # similarity, targetidx = findpossiblematch(checkplaceimg, int(checkplace), range_vec, depthorder_vec,
        #                                           spatial_vec,
        #                                           normal_vec,
        #                                           scancontext_vec,
        #                                           0)
        similarity, targetidx = matchwithyaw(rangemat, scancontextmat, int(checkplace),
                                             0, range_vec, scancontext_vec)
        error_x = pose_new[int(checkplace), 0, 3] - pose_new[targetidx, 0, 3]  
        error_y = pose_new[int(checkplace), 1, 3] - pose_new[targetidx, 1, 3]                                
        locdist = np.sqrt(error_x * error_x + error_y * error_y)

        if similarity < similarity_threshold:                            # Positive Prediction
            if locdist < 3:         
                num_true_positive += 1
                average_sim += similarity
                tp_x.append(pose_new[targetidx, 0, 3])
                tp_y.append(pose_new[targetidx, 1, 3])
            else:
                num_false_positive += 1
                average_sim += similarity
                fp_x.append(pose_new[targetidx, 0, 3])
                fp_y.append(pose_new[targetidx, 1, 3])
        else:
            if locdist < 3:         
                num_false_negative += 1
                average_sim += similarity
                fn_x.append(pose_new[targetidx, 0, 3])
                fn_y.append(pose_new[targetidx, 1, 3])
            else:
                num_true_negative += 1
                average_sim += similarity
                tn_x.append(pose_new[targetidx, 0, 3])
                tn_y.append(pose_new[targetidx, 1, 3])

        print(targetidx)
        min_sim, yawunit = getinitialyaw(scancontextmat[int(checkplace)], scancontextmat[targetidx])
        end = time.time()
        runtime += end - start
        # print("min_sim:", similarity)
        # print("yawunit:", yawunit * 0.4)

    precision = num_true_positive / (num_true_positive + num_false_positive)
    recall = num_true_positive / (num_true_positive + num_false_negative)
    F1_score = 2 * precision * recall / (precision + recall)
    print("average_similarity:", average_sim / len(rangemat))
    print("Query accuracy:", F1_score)
    print("average_runtime", runtime / len(rangemat))
    offset = 5  # meters
    map_size = [min(pose_new[:, 0, 3]) - offset,
                max(pose_new[:, 0, 3]) + offset,
                min(pose_new[:, 1, 3]) - offset,
                max(pose_new[:, 1, 3]) + offset]
    print(pose_new.shape)
    print("x:", pose_new[0, 0, 3])
    fig, ax = plt.subplots()
    # set the map
    ax.set(xlim=map_size[:2], ylim=map_size[2:])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.plot(pose_new[:, 0, 3], pose_new[:, 1, 3], '--', alpha=0.5, c='black')
    ax.plot(tp_x, tp_y, 'ro')
    ax.plot(fp_x, fp_y, 'b+')
    ax.plot(fn_x, fn_y, 'c+')
    # ax.plot(tn_x, tn_y, 'b+')
    # plt.savefig('result.png')
    plt.show()



