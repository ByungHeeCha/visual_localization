from numpy.lib.type_check import imag
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch.utils.data as data
from glob import glob
from PIL import Image
import os
import h5py
import bisect

class Hyundai(data.Dataset):
    def __init__(self, dataset_path, floor, stage, transform=None):
        self.path = sorted(glob(f"{dataset_path}/{floor}/{stage}/*/images/*.jpg"))
        self.lidar_path = sorted(
            glob(f"{dataset_path}/{floor}/{stage}/*/pointclouds_data/*.pcd"))
        self.transform = transform
        self.dataset_path = dataset_path
        self.floor = floor
        self.stage = stage
        hdf_files = sorted(glob(f"{dataset_path}/{floor}/{stage}/*/groundtruth.hdf5"))
        self.pose = []
        self.stamp = []
        self.lidar_pose = []
        self.lidar_stamp = []
        for file in hdf_files:
            with h5py.File(file, 'r') as f:
                for k in sorted(list(f.keys())):
                    if k.startswith('lidar') and k.endswith('pose'):
                        self.lidar_pose = self.lidar_pose + list(f[k])
                    elif k.startswith('lidar') and k.endswith('stamp'):
                        self.lidar_stamp = self.lidar_stamp + list(f[k])
                    elif k.endswith("pose"):
                        self.pose = self.pose + list(f[k])
                    else:
                        self.stamp = self.stamp + list(f[k])
        self.stamp = np.array(self.stamp).squeeze()
        self.pose = np.array(self.pose).squeeze()
        self.lidar_stamp = np.array(self.lidar_stamp).squeeze()
        self.lidar_pose = np.array(self.lidar_pose).squeeze()
        ind = np.argsort(self.lidar_stamp, kind='heapsort')
        self.lidar_stamp = self.lidar_stamp[ind]
        self.lidar_pose = self.lidar_pose[ind]
        self.lidar_path = [self.lidar_path[i] for i in ind]
        assert len(self.pose) == len(self.path) and len(self.stamp) == len(self.path)

    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        img = Image.open(self.path[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
    
    def get_path(self, idx):
        return self.path[idx]
    
    def get_camera_param(self, idx):
        cam_dir_l = self.path[idx].split("/")
        img_rel_path = cam_dir_l[-1]
        cam_dir_l = cam_dir_l[:-2]
        cam_dir_l[0] = "/" + cam_dir_l[0]
        cam_txt_path = os.path.join(*cam_dir_l, "camera_parameters.txt")
        K = None
        raddist = None
        tandist = None
        image_shape = None
        with open(cam_txt_path) as f:
            for i, line in enumerate(f):
                if i<4:
                    continue
                cam_id = img_rel_path.split("_")[0]
                if cam_id == line.split(" ")[0]:
                    line = line.strip("\n")
                    param = line.split(" ")
                    K = np.asarray([[param[3], 0, param[5]], [0, param[4], param[6]], [0, 0, 1]]).astype("float")
                    raddist = np.array([param[7], param[8], param[11]]).astype("float")
                    tandist = np.array([param[9], param[10]]).astype("float")
                    image_shape = (int(param[2]), int(param[1]))

                    break
        if K is None:
            raise Exception("No valid camera parameter")
        return K, raddist, tandist, image_shape
    
    def lidar_project(self, idx):
        ts = self.stamp[idx]
        low = ts - 4*1000000
        high = ts + 4*1000000
        low_ind = np.searchsorted(self.lidar_stamp, low, side='left')
        high_ind = np.searchsorted(self.lidar_stamp, high, side='right')
        # print(ts, self.lidar_stamp[low_ind+1], self.lidar_stamp[high_ind-100])
        return self.lidar_pose[low_ind:high_ind], self.lidar_path[low_ind:high_ind]
    
    def get_pose(self, idx):
        return self.pose[idx]




class HyundaiTest(data.Dataset):
    def __init__(self, dataset_path, floor, stage, transform=None):
        self.path = sorted(glob(f"{dataset_path}/{floor}/{stage}/*/images/*.jpg"))
        self.transform = transform
        self.dataset_path = dataset_path
        self.floor = floor
        self.stage = stage

    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        img = Image.open(self.path[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
    
    def get_path(self, idx):
        return self.path[idx]
    
    def get_camera_param(self, idx):
        cam_dir_l = self.path[idx].split("/")
        img_rel_path = cam_dir_l[-1]
        cam_dir_l = cam_dir_l[:-2]
        cam_dir_l[0] = "/" + cam_dir_l[0]
        cam_txt_path = os.path.join(*cam_dir_l, "camera_parameters.txt")
        K = None
        raddist = None
        tandist = None
        image_shape = None
        with open(cam_txt_path) as f:
            for i, line in enumerate(f):
                if i<4:
                    continue
                cam_id = img_rel_path.split("_")[0]

                if cam_id == line.split(" ")[0]:
                    line = line.strip("\n")
                    param = line.split(" ")
                    K = np.asarray([[param[3], 0, param[5]], [0, param[4], param[6]], [0, 0, 1]]).astype("float")
                    raddist = np.array([param[7], param[8], param[11]]).astype("float")
                    tandist = np.array([param[9], param[10]]).astype("float")
                    image_shape = (int(param[2]), int(param[1]))

                    break
        if K is None:
            raise Exception("No valid camera parameter")
        return K, raddist, tandist, image_shape
    
    # def project(self, idx):
    #     img_rel_path = self.path[idx].split("/")[-1]
    #     img_filename = img_rel_path.split(".")[0]
    #     timestamp = int(img_filename.split("_")[1])
    #     date = datetime.datetime.fromtimestamp(timestamp)
    #     lower = -1
    #     upper = len(self.path)
    #     for i in range(idx-1, -1, -1):
    #         img_rel_path = self.path[i].split("/")[-1]
    #         img_filename = img_rel_path.split(".")[0]
    #         timestamp = int(img_filename.split("_")[1])
    #         if abs(datetime.datetime.fromtimestamp(timestamp) - date).total_seconds() > 4:
    #             lower = i
    #             break
        
    #     for i in range(idx+1, len(self.path)):
    #         img_rel_path = self.path[i].split("/")[-1]
    #         img_filename = img_rel_path.split(".")[0]
    #         timestamp = int(img_filename.split("_")[1])
    #         if abs(datetime.datetime.fromtimestamp(timestamp) - date).total_seconds() > 4:
    #             upper = i
    #             break
    #     return range(lower+1, upper)

