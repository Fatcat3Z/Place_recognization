import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.spatial as spt
import os
from tqdm import tqdm
import sys
import time
import math
from scipy.spatial.distance import cdist
from img_gist_feature.utils_gist import *
from img_gist_feature.util__base import *
from img_gist_feature.util__cal import *


def eucliDist(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


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


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


class LoopClosureDetect:
    def __init__(self, imgfilepaths, mode):
        self.mode = mode
        self.tree_making_period_conter = 0
        self.imgfilepaths = imgfilepaths
        self.rangemat = []
        self.spatialmat = []
        self.mergespatialmat = []
        self.normalmat = []
        self.scancontextmat = []
        self.featuremat = []
        self.gistfeature = []
        self.rangegist = []
        self.scancontextgist = []
        self.spatialgist = []
        self.NUM_EXCLUDE_RECENT = 25
        self.TREE_MAKING_PERIOD = 1
        self.sector = 900
        self.ring = 64
        # self.serachkdtree = spt.KDTree(data=self.sawsenses, leafsize=10)
        self.ringkeys = []  # all ringkeys for new map, use to generate tree

    def getfilenum(self, filepath):
        files = os.listdir(filepath)
        return len(files)

    def gettargetimgraw(self, filepath, idx, colormode, feature):
        if colormode == "gray":
            img = cv2.imread(filepath + idx + ".jpg", -1)
        else:
            img = cv2.imread(filepath + idx + ".jpg")

        if feature == "range":
            self.rangemat.append(img)
        elif feature == "spatial":
            self.spatialmat.append(img)
        elif feature == "normal":
            self.normalmat.append(img)
        elif feature == "scancontext":
            self.scancontextmat.append(img)
        # return img

    def gettargetimg(self, filepath, idx, colormode, feature):
        if colormode == "gray":
            img = cv2.imread(filepath + idx + ".jpg", -1)
            arrayimg = np.array(img, dtype=int)
        else:
            img = cv2.imread(filepath + idx + ".jpg")
            arrayimg = np.array(img, dtype=int)
        if feature == "range":
            self.rangemat.append(arrayimg)
        elif feature == "spatial":
            self.spatialmat.append(arrayimg)
        elif feature == "normal":
            self.normalmat.append(arrayimg)
        elif feature == "scancontext":
            self.scancontextmat.append(arrayimg)
        # return arrayimg

    def fastalignusingvkey(self, vkey1, vkey2):
        argmin_vkey_shift = 0
        vkey_diff = [a - b for a, b in zip(vkey1, vkey2)]
        min_veky_diff_norm = np.linalg.norm(vkey_diff)
        interval = 3
        steps = int(len(vkey1) / interval)
        for i in range(steps):
            vkey2 = np.roll(vkey2, interval, axis=0)
            vkey_diff = [a - b for a, b in zip(vkey1, vkey2)]
            cur_diff = np.linalg.norm(vkey_diff)
            if cur_diff < min_veky_diff_norm:
                argmin_vkey_shift = i * 3 + 1
        return argmin_vkey_shift

    def calculatesim_dim(self, img1, img2):
        num_sectors = self.sector
        vkey1 = self.getSectorKey(img1)
        vkey2 = self.getSectorKey(img2)
        shift_cols = self.fastalignusingvkey(vkey1, vkey2)
        sum_of_cos_sim_raw = 0
        num_col_engaged_raw = 0
        sum_of_cos_sim_shift = 0
        num_col_engaged_shift = 0

        for j in range(num_sectors):
            col_j_1 = img1[:, j]
            col_j_2 = img2[:, j]
            if ~np.any(col_j_1) or ~np.any(col_j_2):
                continue
            # calc sim
            cos_similarity = np.dot(col_j_1, col_j_2) / (np.linalg.norm(col_j_1) * np.linalg.norm(col_j_2))
            sum_of_cos_sim_raw = sum_of_cos_sim_raw + cos_similarity
            num_col_engaged_raw = num_col_engaged_raw + 1

        if num_col_engaged_raw == 0:
            sim = 0
        else:
            sim = sum_of_cos_sim_raw / num_col_engaged_raw
        # print(sim)
        if shift_cols != 0:
            shifted_img2 = np.roll(img1, shift_cols, axis=1)
            for j in range(num_sectors):
                col_j_1 = img1[:, j]
                col_j_2 = shifted_img2[:, j]
                if ~np.any(col_j_1) or ~np.any(col_j_2):
                    continue
                # calc sim
                cos_similarity = np.dot(col_j_1, col_j_2) / (np.linalg.norm(col_j_1) * np.linalg.norm(col_j_2))
                sum_of_cos_sim_shift = sum_of_cos_sim_shift + cos_similarity
                num_col_engaged_shift = num_col_engaged_shift + 1

            if num_col_engaged_shift == 0:
                sim_shift = 0
            else:
                sim_shift = sum_of_cos_sim_shift / num_col_engaged_shift
            # print(sim_shift)
            if sim_shift > sim:
                sim = sim_shift

        return sim

    def calculate_cossim(self, vkey1, vkey2):
        shift_cols = self.fastalignusingvkey(vkey1, vkey2)
        sum_of_cos_sim_raw = 0
        num_col_engaged_raw = 0
        sum_of_cos_sim_shift = 0
        num_col_engaged_shift = 0

        cos_similarity = np.dot(vkey1, vkey2) / (np.linalg.norm(vkey1) * np.linalg.norm(vkey2))

        return cos_similarity

    def pro_distance(self, sc1, sc2):
        if sc1.ndim == 2:
            return self.calculatesim_dim(sc1, sc2)
        elif sc1.ndim == 3:
            mat0_0 = sc1[:, :, 0]
            mat0_1 = sc1[:, :, 1]
            mat0_2 = sc1[:, :, 2]
            mat1_0 = sc2[:, :, 0]
            mat1_1 = sc2[:, :, 1]
            mat1_2 = sc2[:, :, 2]
            sim1 = self.calculatesim_dim(mat0_0, mat1_0)
            sim2 = self.calculatesim_dim(mat0_1, mat1_1)
            sim3 = self.calculatesim_dim(mat0_2, mat1_2)
            return (sim1 + sim2 + sim3) / 3

    def getRingKey(self, img):

        ringkey = []
        for i in range(64):
            ringkey.append(np.mean(img[i]))
        return ringkey

    def getSectorKey(self, img):
        sectorkey = []
        for i in range(self.sector):
            sectorkey.append(np.mean(img[:, i]))

        return sectorkey

    def merge_spatialfeature(self, spatialmat, rangemat):
        mergedfeature = np.zeros((64, 900), np.uint8)
        for i in range(64):
            for j in range(900):
                # if spatialmat[i][j] == 0:
                #     mergedfeature[i][j] = rangemat[i][j]
                # else:
                #     mergedfeature[i][j] = spatialmat[i][j]
                values = [int(rangemat[i][j]), int(spatialmat[i][j])]
                merge_weight = softmax(values)
                mergedfeature[i][j] = merge_weight[0] * rangemat[i][j] + merge_weight[1] * spatialmat[i][j]
        # print(mergedfeature)
        return mergedfeature

    def generatenewmap(self, cur_idx):
        if self.mode == 0:
            self.gettargetimg(self.imgfilepaths[0], cur_idx, colormode="gray", feature="range")
            self.featuremat.append(self.rangemat[-1])
            self.ringkeys.append(self.getRingKey(self.rangemat[-1]))
        elif self.mode == 1:
            self.gettargetimg(self.imgfilepaths[2], cur_idx, colormode="gray", feature="scancontext")
            self.featuremat.append(self.scancontextmat[-1])
            self.ringkeys.append(self.getRingKey(self.scancontextmat[-1]))
        elif self.mode == 2:
            self.gettargetimg(self.imgfilepaths[0], cur_idx, colormode="gray", feature="range")
            self.gettargetimg(self.imgfilepaths[2], cur_idx, colormode="gray", feature="scancontext")

    def generatenewmapgist(self, cur_idx):
        if self.mode == 0:
            self.gettargetimgraw(self.imgfilepaths[0], cur_idx, colormode="gray", feature="range")
            self.gistfeature.append(get_img_gist_feat(self.rangemat[-1], "gray")[0])
            self.featuremat.append(self.rangemat[-1])
        elif self.mode == 1:
            self.gettargetimgraw(self.imgfilepaths[2], cur_idx, colormode="gray", feature="scancontext")
            self.gistfeature.append(get_img_gist_feat(self.scancontextmat[-1], "gray")[0])
            self.featuremat.append(self.scancontextmat[-1])
        elif self.mode == 2:
            self.gettargetimgraw(self.imgfilepaths[0], cur_idx, colormode="gray", feature="range")
            self.gettargetimgraw(self.imgfilepaths[2], cur_idx, colormode="gray", feature="scancontext")
            self.rangegist.append(get_img_gist_feat(self.rangemat[-1], "gray")[0])
            self.scancontextgist.append(get_img_gist_feat(self.scancontextmat[-1], "gray")[0])
            # gistrange = get_img_gist_feat(self.rangemat[-1], "gray")[0]
            # gistscancontext = get_img_gist_feat(self.scancontextmat[-1], "gray")[0]
            # merged_gist = []
            # for i in range(len(gistrange)):
            #     rawgist = [gistrange[i], gistscancontext[i]]
            #     merged_wights = softmax(rawgist)
            #     merged_gist.append(merged_wights[0] * gistrange[i] + merged_wights[1] * gistscancontext[i])
            # self.gistfeature.append(merged_gist)
        elif self.mode == 3:
            self.gettargetimgraw(self.imgfilepaths[1], cur_idx, colormode="gray", feature="normal")
            self.gistfeature.append(get_img_gist_feat(self.normalmat[-1], "gray")[0])
            self.featuremat.append(self.normalmat[-1])
        elif self.mode == 4:
            self.gettargetimgraw(self.imgfilepaths[0], cur_idx, colormode="gray", feature="range")
            self.gettargetimgraw(self.imgfilepaths[3], cur_idx, colormode="gray", feature="spatial")
            merged_img = self.merge_spatialfeature(self.spatialmat[-1], self.rangemat[-1])
            gistdatamerge = get_img_gist_feat(merged_img, "gray")[0]
            self.mergespatialmat.append(merged_img)
            self.gistfeature.append(gistdatamerge)
        elif self.mode == 5:
            self.gettargetimgraw(self.imgfilepaths[0], cur_idx, colormode="gray", feature="range")
            self.gettargetimgraw(self.imgfilepaths[4], cur_idx, colormode="gray", feature="spatial")
            merged_img = self.merge_spatialfeature(self.spatialmat[-1], self.rangemat[-1])
            gistdatamerge = get_img_gist_feat(merged_img, "gray")[0]
            self.mergespatialmat.append(merged_img)
            self.gistfeature.append(gistdatamerge)

            # feature merge
            # gistrange = get_img_gist_feat(self.rangemat[-1], "gray")[0]
            # gistspatial = get_img_gist_feat(merged_img, "gray")[0]
            # merged_gist = []
            # for i in range(len(gistrange)):
            #     rawgist = [gistrange[i], gistspatial[i]]
            #     merged_wights = softmax(rawgist)
            #     merged_gist.append(merged_wights[0] * gistrange[i] + merged_wights[1] * gistspatial[i])

    def lpcdetect(self):
        targetidx = -1
        if len(self.ringkeys) < self.NUM_EXCLUDE_RECENT + 1:
            return targetidx, 0.0
        curr_key = self.ringkeys[-1]
        curr_rangemat = self.featuremat[-1]

        if self.tree_making_period_conter % self.TREE_MAKING_PERIOD == 0:
            serachdata = self.ringkeys[0: -self.NUM_EXCLUDE_RECENT]
            print("search len: ", len(serachdata))
            self.kdtree = spt.KDTree(data=serachdata, leafsize=64)
        self.tree_making_period_conter = self.tree_making_period_conter + 1

        _, queidx = self.kdtree.query(curr_key, 10)
        # print("cur queidx", queidx)
        max_sim = 0
        for i in range(len(queidx)):
            targetmat = self.featuremat[queidx[i]]
            similarty = self.pro_distance(curr_rangemat, targetmat)
            if similarty > max_sim:
                max_sim = similarty
                targetidx = queidx[i]

        return targetidx, max_sim

    def lpcdetectgist(self):
        targetidx = -1
        if self.mode == 0 or self.mode == 1 or self.mode == 3 or self.mode == 4 or self.mode == 5:
            if len(self.gistfeature) < self.NUM_EXCLUDE_RECENT + 1:
                return targetidx, 0.0
            # print(len(self.gistfeature))
            curr_key = self.gistfeature[-1]
            # curr_merged = self.mergespatialmat[-1]
            # curr_rangemat = self.featuremat[-1]

            if self.tree_making_period_conter % self.TREE_MAKING_PERIOD == 0:
                serachdata = self.gistfeature[0: -self.NUM_EXCLUDE_RECENT]
                print("search len: ", len(serachdata))
                self.kdtree_sta = spt.KDTree(data=serachdata, leafsize=10)
                # self.kdtree_dyn = spt.KDTree(data=serachdata, leafsize=64)
            self.tree_making_period_conter = self.tree_making_period_conter + 1

            distance_sta, queidx_sta = self.kdtree_sta.query(curr_key, 5)
            # distance_dyn, queidx_dyn = self.kdtree_dyn.query(curr_key, 5)
            print("cur queidx", queidx_sta)
            max_sim = distance_sta[0]
            targetidx = queidx_sta[0]

            # for i in range(len(queidx_sta)):
            #     sim = self.calculatesim_dim(curr_merged, self.mergespatialmat[queidx_sta[i]])
            #     if sim > max_sim:
            #         max_sim = sim
            #         targetidx = queidx_sta[i]

            # print(self.featuremat[targetidx])
            # max_sim = self.pro_distance(self.featuremat[targetidx], self.featuremat[-1])


        else:
            if len(self.rangegist) < self.NUM_EXCLUDE_RECENT + 1:
                return targetidx, 0.0
            cur_rangegist = self.rangegist[-1]
            cur_scancontextgist = self.scancontextgist[-1]
            # build different trees
            if self.tree_making_period_conter % self.TREE_MAKING_PERIOD == 0:
                serachrangedata = self.rangegist[0: -self.NUM_EXCLUDE_RECENT]
                serachscandata = self.scancontextgist[0: -self.NUM_EXCLUDE_RECENT]
                self.kdtree_sta = spt.KDTree(data=serachrangedata, leafsize=10)
                self.kdtree_scan = spt.KDTree(data=serachscandata, leafsize=10)
            self.tree_making_period_conter = self.tree_making_period_conter + 1

            distance_range, queidx_range = self.kdtree_sta.query(cur_rangegist, 5)
            distance_scan, queidx_scan = self.kdtree_scan.query(cur_scancontextgist, 5)

            range_first_idx = -1
            scan_first_idx = -1
            max_sim = 1
            for i in range(len(queidx_range)):
                if queidx_range[i] in queidx_scan:
                    range_first_idx = i
                    break
            for i in range(len(queidx_scan)):
                if queidx_scan[i] in queidx_range:
                    scan_first_idx = i
                    break
            # if both can not find the suitable idx, choose the latest one
            if range_first_idx == -1:
                range_first_idx = 0
            if scan_first_idx == -1:
                scan_first_idx = 0

            if queidx_range[range_first_idx] == queidx_scan[scan_first_idx]:
                disweights = [1 / distance_range[range_first_idx], 1 / distance_scan[range_first_idx]]
                outweights = softmax(disweights)
                targetidx = queidx_range[range_first_idx]
                max_sim = outweights[0] * distance_range[range_first_idx] + outweights[1] * distance_scan[
                    range_first_idx]
            else:
                tartget_scanidx = np.where(queidx_scan == queidx_range[range_first_idx])
                # print(tartget_scanidx)
                # print(len(tartget_scanidx[0]))
                if len(tartget_scanidx[0]) > 0:
                    range_first_weights = [1 / distance_range[range_first_idx],
                                           1 / distance_scan[tartget_scanidx[0][0]]]
                    range_first_outweights = softmax(range_first_weights)
                    range_first_sim = range_first_outweights[0] * distance_range[range_first_idx] + \
                                      range_first_outweights[1] * distance_scan[tartget_scanidx]
                else:
                    range_first_sim = distance_range[range_first_idx]

                tartget_rangeidx = np.where(queidx_range == queidx_scan[scan_first_idx])
                if len(tartget_scanidx[0]) > 0:
                    scan_first_weights = [1 / distance_range[tartget_rangeidx[0][0]], 1 / distance_scan[scan_first_idx]]
                    scan_first_outweights = softmax(scan_first_weights)
                    scan_first_sim = scan_first_outweights[0] * distance_range[tartget_rangeidx] + \
                                     scan_first_outweights[1] * distance_scan[scan_first_idx]
                else:
                    scan_first_sim = distance_scan[scan_first_idx]

                targetidx = queidx_range[range_first_idx] if range_first_sim < scan_first_sim else queidx_scan[
                    scan_first_idx]
                max_sim = range_first_sim if range_first_sim < scan_first_sim else scan_first_sim

        # choose static or dynamic
        # if distance_sta[0] < distance_dyn[0]:
        #     targetidx = queidx_sta[0]
        #     max_sim = distance_sta[0]
        # else:
        #     targetidx = queidx_dyn[0]
        #     max_sim = distance_dyn[0]

        # for i in range(len(queidx)):
        #     targetvec = self.gistfeature[queidx[i]]
        #     euc_sim = 1 - distance[i]
        #     cos_sim = self.calculate_cossim(curr_key, targetvec)
        #     aver_sim = (euc_sim + cos_sim) / 2
        #     # similarty = self.pro_distance(curr_rangemat, targetmat)
        #     print("euc_dis: ", euc_sim)
        #     print("cos_sim: ", cos_sim)
        #     print(aver_sim)
        #     if aver_sim > max_sim:
        #         max_sim = aver_sim
        #         targetidx = queidx[i]

        return targetidx, max_sim


if __name__ == "__main__":
    rootpath = "/home/wy-lab/kitti_odometry_dataset/sequences/"
    sequence = sys.argv[1]
    imgmode = "SegImgs"
    rangefile = rootpath + sequence + "/" + imgmode + "/RangeMat/"
    spatialfile = rootpath + sequence + "/" + imgmode + "/SpatialMat/"
    normalfile = rootpath + sequence + "/" + imgmode + "/NormalMat/"
    scanconetxtfile = rootpath + sequence + "/" + imgmode + "/ScanContextMat/"
    spatialsctfile = rootpath + sequence + "/" + imgmode + "/SpatialSCTMat/"
    poses_file = rootpath + sequence + "/" + sequence + ".txt"
    calib_file = rootpath + sequence + "/calib.txt"

    rawmode = "RawImgs"
    rawrange = rootpath + sequence + "/" + rawmode + "/RangeMat/"
    rawscancontext = rootpath + sequence + "/" + rawmode + "/ScanContextMat/"
    rawnormal = rootpath + sequence + "/" + rawmode + "/NormalMat/"

    pose_new = calculate_poses(poses_file, calib_file)
    print(rangefile)
    featurepaths = [rangefile, spatialfile, normalfile, scanconetxtfile]
    rawpaths = [rawrange, rawnormal, rawscancontext, spatialfile, spatialsctfile]
    selectmode = int(sys.argv[2])

    lpd = LoopClosureDetect(rawpaths, selectmode)
    inputnums = lpd.getfilenum(featurepaths[0])
    print("inputnums: ", inputnums)
    calculate_method = sys.argv[3]
    # thresholds = [0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.18, 0.19, 0.195, 0.2, 0.205, 0.21, 0.211, 0.212, 0.213, 0.214, 0.215, 0.216,
    #               0.217, 0.218, 0.219, 0.22, 0.225, 0.23, 0.24, 0.25, 0.26, 0.27]
    thresholds = [0.6, 0.65, 0.7, 0.75, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9]
    num_true_positive = [0] * len(thresholds)
    num_false_positive = [0] * len(thresholds)
    num_true_negative = [0] * len(thresholds)
    num_false_negative = [0] * len(thresholds)
    average_sim = [0] * len(thresholds)
    runtime = 0

    tp_x = [[]] * len(thresholds)
    tp_y = [[]] * len(thresholds)
    fp_x = [[]] * len(thresholds)
    fp_y = [[]] * len(thresholds)
    tn_x = [[]] * len(thresholds)
    tn_y = [[]] * len(thresholds)
    fn_x = [[]] * len(thresholds)
    fn_y = [[]] * len(thresholds)
    # print(tp_x)
    # tp_x[1] = tp_x[1] + [1]
    # print(tp_x)
    for i in tqdm(range(1, inputnums)):
        cur_mat_idx = str(i).rjust(4, "0")
        start = time.time()
        lpd.generatenewmap(cur_mat_idx)
        target_idx, similarity = lpd.lpcdetect()
        # lpd.generatenewmapgist(cur_mat_idx)
        # print(lpd.gistfeature[-1])
        # target_idx, similarity = lpd.lpcdetectgist()
        print("similarity", similarity)
        end = time.time()
        runtime += end - start
        if target_idx == -1:
            continue
        error_x = pose_new[int(cur_mat_idx), 0, 3] - pose_new[target_idx, 0, 3]
        error_y = pose_new[int(cur_mat_idx), 1, 3] - pose_new[target_idx, 1, 3]
        locdist = np.sqrt(error_x * error_x + error_y * error_y)
        dist_threshold = 3
        for similarity_th in range(0, len(thresholds)):
            # euc dist: <
            # cosin dist: >
            if similarity > thresholds[similarity_th]:  # Positive Prediction
                if locdist < dist_threshold:
                    num_true_positive[similarity_th] += 1
                    average_sim[similarity_th] += similarity
                    # tp_x[similarity_th].append(pose_new[target_idx, 0, 3])
                    # tp_y[similarity_th].append(pose_new[target_idx, 1, 3])
                    tp_x[similarity_th] = tp_x[similarity_th] + [pose_new[target_idx, 0, 3]]
                    tp_y[similarity_th] = tp_y[similarity_th] + [pose_new[target_idx, 1, 3]]
                    # print(target_idx)
                else:
                    num_false_positive[similarity_th] += 1
                    average_sim[similarity_th] += similarity
                    # print("FP similarity", similarity)
                    # fp_x[similarity_th].append(pose_new[target_idx, 0, 3])
                    # fp_y[similarity_th].append(pose_new[target_idx, 1, 3])
                    fp_x[similarity_th] = tp_x[similarity_th] + [pose_new[target_idx, 0, 3]]
                    fp_y[similarity_th] = tp_y[similarity_th] + [pose_new[target_idx, 1, 3]]
            else:
                if locdist < dist_threshold:
                    num_false_negative[similarity_th] += 1
                    average_sim[similarity_th] += similarity
                    # print("FN similarity", similarity)
                    # fn_x[similarity_th].append(pose_new[target_idx, 0, 3])
                    # fn_y[similarity_th].append(pose_new[target_idx, 1, 3])
                    fn_x[similarity_th] = tp_x[similarity_th] + [pose_new[target_idx, 0, 3]]
                    fn_y[similarity_th] = tp_y[similarity_th] + [pose_new[target_idx, 1, 3]]
                else:
                    num_true_negative[similarity_th] += 1
                    average_sim[similarity_th] += similarity
                    # tn_x[similarity_th].append(pose_new[target_idx, 0, 3])
                    # tn_y[similarity_th].append(pose_new[target_idx, 1, 3])
                    tn_x[similarity_th] = tp_x[similarity_th] + [pose_new[target_idx, 0, 3]]
                    tn_y[similarity_th] = tp_y[similarity_th] + [pose_new[target_idx, 1, 3]]

    modecontent = ["range",  # 0
                   "scancontext",  # 1
                   "range_scancontext",  # 2
                   "normal",  # 3
                   "range_spatial",  # 4
                   "scancontext_spatial",  # 5
                   "scancontext_normal",  # 6
                   "range_scancontext_spatial",  # 7
                   "range_scancontext_normal",  # 8
                   "range_scancontext_spatial_normal"]  # 9
    maxF1score = 0
    maxthreshold = 0
    maxthresholdidx = 0
    finalprecision = 0
    finalrecall = 0
    precision = 0
    recall = 0
    for i in range(0, len(thresholds)):
        if num_true_positive[i] + num_false_positive[i] == 0 or num_true_positive[i] + num_false_negative[i] == 0 or \
                num_true_positive[i] == 0:
            continue
        precision = num_true_positive[i] / (num_true_positive[i] + num_false_positive[i])
        recall = num_true_positive[i] / (num_true_positive[i] + num_false_negative[i])
        F1_score = 2 * precision * recall / (precision + recall)
        finalprecision = precision if F1_score > maxF1score else finalprecision
        finalrecall = recall if F1_score > maxF1score else finalrecall
        maxthresholdidx = i if F1_score > maxF1score else maxthresholdidx
        maxF1score = F1_score if F1_score > maxF1score else maxF1score
    # print("TP num: ", num_true_positive[maxthresholdidx])
    # print("FP num: ", num_false_positive[maxthresholdidx])
    print(modecontent[selectmode] + " average_similarity: ", average_sim[maxthresholdidx] / inputnums)
    print(modecontent[selectmode] + " max threshold choose: ", thresholds[maxthresholdidx])
    print(modecontent[selectmode] + " Precition: ", finalprecision)
    print(modecontent[selectmode] + " Recall: ", finalrecall)
    print(modecontent[selectmode] + " Query accuracy: ", maxF1score)
    print(modecontent[selectmode] + " average_runtime： ", runtime / inputnums)
    offset = 5  # meters
    map_size = [min(pose_new[:, 0, 3]) - offset,
                max(pose_new[:, 0, 3]) + offset,
                min(pose_new[:, 1, 3]) - offset,
                max(pose_new[:, 1, 3]) + offset]
    # print(pose_new.shape)
    # print("x:", pose_new[0, 0, 3])
    fig, ax = plt.subplots()
    # set the map
    plt.suptitle(sequence + " " + calculate_method + " " + modecontent[selectmode] + ' result', fontsize=15)
    ax.set(xlim=map_size[:2], ylim=map_size[2:])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title("t:  " + '%.3f' % thresholds[maxthresholdidx] + \
                 "  F1maxScore:  " + '%.4f' % maxF1score + \
                 "  Precision:  " + '%.4f' % finalprecision + \
                 "  Recall:  " + '%.4f' % finalrecall, fontsize=12)
    ax.plot(pose_new[:, 0, 3], pose_new[:, 1, 3], '--', alpha=0.5, c='black', label='trajectory')

    ax.plot(tp_x[maxthresholdidx], tp_y[maxthresholdidx], 'ro', label='true positive')
    ax.plot(fp_x[maxthresholdidx], fp_y[maxthresholdidx], 'b+', label='false positive')
    ax.plot(fn_x[maxthresholdidx], fn_y[maxthresholdidx], 'c+', label='false negative')
    leg = ax.legend()
    plt.savefig(sequence + "_Raw_ " + modecontent[selectmode] + "_" + calculate_method + '_result' + '.png')
