from kornia.feature import responses
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.transforms import Grayscale
from tqdm import tqdm
from torchvision import transforms
from data import Hyundai, HyundaiTest
from models import NetVLAD
import cv2
import faiss
import kornia
from torch.utils.data import DataLoader
import pickle
from pyquaternion import Quaternion
from SuperGluePretrainedNetwork import models
from utils import pcl_utils
import cupy as cp
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--floor', type=str, help='floor: b1, 1f')
parser.add_argument('--globaldesc', type=str, help='Global descriptor: netvlad, apgem')
parser.add_argument('--rank_knn_num', type=int, default=10,
                    help='number of nearest neighbor to find per query image on ranking phase')
parser.add_argument('--rerank_knn_num', type=int, default=5,
                    help='number of nearest neighbor to find per query image on reranking phase')
parser.add_argument('--dataset_path', type=str, help='Path of a dataset', default="../dataset")

args = parser.parse_args()

dimension = {'netvlad': 32768, 'apgem': 2048}
# import pcl

# FLANN_INDEX_LSH = 6
FLANN_INDEX_KDTREE = 0


def resize(img):
    w, h = img.size
    return transforms.Resize((h//2, w//2))(img)

def tocv(img):
    return np.array(img)


def input_transform():
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])


def ranking(query_dict, db_dict):
    # query_dict = query_dict.cuda()
    # db_dict = db_dict.cuda()
    # print(query_dict.shape)
    faiss_index = faiss.IndexFlatL2(dimension[args.globaldesc])
    res = faiss.StandardGpuResources()
    faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
    faiss_index.add(db_dict)
    _, indices = faiss_index.search(query_dict, args.rank_knn_num)
    return indices

def local_match(neighbor, q_dataset, db_dataset):
    # rootsift = RootSIFT()
    pbar = tqdm(neighbor)
    reranked_neigh = []
    total_match = []
    print("Start reranking...")
    superPointdict=dict()
    superPointdict["nms_radius"]=4
    superPointdict["keypoint_threshold"]= 0.005
    superPointdict["max_keypoints"]=1024
    superpoint = models.SuperPoint(superPointdict)
    superpoint.cuda()
    superpoint.eval()
    superGluedict=dict()
    superGluedict["weights"]="indoor"
    superGluedict["sinkhorn_iterations"]= 20
    superGluedict["match_threshold"]=0.02
    superglue = models.SuperGlue(superGluedict)
    superglue.cuda()
    superglue.eval()
    for q, indices in enumerate(pbar):
        if q >= 101: break
        q_img = q_dataset[q].cuda()
        with torch.no_grad():
            featq = superpoint({'image': q_img.unsqueeze(0)})
        kpq = featq['keypoints']
        descq = featq['descriptors']
        scoreq = featq['scores']
        q_match = []
        db_match = []
        num_inlier = []
        for ind in indices:
            db_img = db_dataset[ind].cuda()
            with torch.no_grad():
                featdb = superpoint({'image': db_img.unsqueeze(0)})
            kpdb = featdb['keypoints']
            descdb = featdb['descriptors']
            scoredb = featdb['scores']
            
            data = {"image0": q_img.unsqueeze(
                0), "image1": db_img.unsqueeze(0)}

            pred = {}
            pred0 = {"keypoints": kpq,
                     "descriptors": descq,
                     "scores": scoreq}
            pred1 = {"keypoints": kpdb,
                     "descriptors": descdb,
                     "scores": scoredb}
            
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

            data = {**data, **pred}

            for k in data:
                if isinstance(data[k], (list, tuple)):
                    data[k] = torch.stack(data[k])
            pred = {**pred, **superglue(data)}

            pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
            matches = pred['matches0']

            valid = matches > -1
            conf = pred['matching_scores0'][valid]
            
            points_q = kpq[0].cpu().detach().numpy()[valid]
            points_db = kpdb[0].cpu().detach().numpy()[matches[valid]]
            q_match.append(points_q)
            db_match.append(points_db)
            num_inlier.append(points_q.shape[0])
        num_inlier = np.array(num_inlier)
        sort_ind = num_inlier.argsort()[::-1]
        reranked_neigh.append(indices[sort_ind][:args.rerank_knn_num])
        q_match = [q_match[i] for i in sort_ind[:args.rerank_knn_num]]
        db_match = [db_match[i] for i in sort_ind[:args.rerank_knn_num]]
        total_match.append((q_match, db_match))
    reranked_neigh = np.stack(reranked_neigh)
    return reranked_neigh, total_match

def pose_estimation(reranked_neigh, total_match, q_dataset, db_dataset, resized_sh):
    pbar = tqdm(reranked_neigh)
    print("Start pose estimation")
    num_strange = 0
    ans = []
    for q, db_index in enumerate(pbar):
        q_K, q_rot, q_tan, q_sh = q_dataset.get_camera_param(q)
        # q_tf_mat = 
        q_pairs = []
        pair_3d =[]
        for i, ind in enumerate(db_index):
            db_K, db_rot, db_tran, db_sh = db_dataset.get_camera_param(ind)
            lidar_pose, lidar_path = db_dataset.lidar_project(ind)
            p = db_dataset.get_pose(ind)
            pose = cp.eye(4)
            pose[:3, 3] = cp.asarray(p[:3])
            pose[:3, :3] = cp.asarray(Quaternion(p[3:]).rotation_matrix)
            points_3d = []
            for lp, lpath in zip(lidar_pose, lidar_path):
                lpose = cp.eye(4)
                lpose[:3, 3] = cp.asarray(lp[:3])
                lpose[:3, :3] = cp.asarray(Quaternion(lp[3:]).rotation_matrix)
                points_3d_local = cp.asarray(pcl_utils.load_pointcloud(lpath))
                points_3d_local = cp.concatenate([points_3d_local, cp.ones((points_3d_local.shape[0], 1))], axis=1)
                points_3d.append(points_3d_local @ lpose.T)
            points_3d = cp.concatenate(points_3d)
            points_2d = (points_3d @ cp.linalg.inv(pose).T)[:, :3] @ cp.asarray(db_K).T
            points_2d = (points_2d / points_2d[:, 2:3])[:, :2]
            points_2d = cp.asnumpy(points_2d).astype(np.float32)
            faiss_index = faiss.IndexFlatL2(2)
            res = faiss.StandardGpuResources()
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
            faiss_index.add(points_2d)
            a = total_match[q][1][i]
            if a.shape[0] == 0:
                a = np.zeros((0, 2))
            a[:, 0] = a[:, 0] * db_sh[1] / resized_sh[1]
            a[:, 1] = a[:, 1] * db_sh[0] / resized_sh[0]
            vals, indices = faiss_index.search(a.astype(np.float32), 1)
            del faiss_index
            vals = vals.squeeze(1)
            indices = indices.squeeze(1)
            indices = indices[vals < 7.5]
            points_3d = points_3d[indices]
            b = total_match[q][0][i]
            if b.shape[0] == 0:
                b = np.zeros((0, 2))
            b[:, 0] = b[:, 0] * q_sh[1] / resized_sh[1]
            b[:, 1] = b[:, 1] * q_sh[0] / resized_sh[0]
            q_pairs.append(b[vals < 7.5])
            pair_3d.append(points_3d)
        q_pairs = np.concatenate(q_pairs)
        pair_3d = cp.concatenate(pair_3d)[:, :3]
        try:
            _, solvR, solvt, inlierR = cv2.solvePnPRansac(np.ascontiguousarray(cp.asnumpy(pair_3d)).reshape((-1, 3)),
                                                    np.ascontiguousarray(q_pairs).reshape((-1, 2)), \
                                                        np.ascontiguousarray(q_K.astype(
                                                            np.float32)).reshape((3, 3)),
                                                    np.array([q_rot[0], q_rot[1], q_tan[0], q_tan[1], q_rot[2]]).astype(np.float32), \
                                                    # np.array([0, 0, 0, 0, 0]).astype(np.float32), \
                                                    iterationsCount=100000, \
                                                    useExtrinsicGuess = True, \
                                                    confidence = 0.999, \
                                                    reprojectionError = 8, \
                                                    flags = cv2.SOLVEPNP_AP3P)
            solvRR,_ = cv2.Rodrigues(solvR)
            solvRR_inv = np.linalg.inv(solvRR)
            solvtt = -np.matmul(solvRR_inv,solvt)
            
            rot = cv2.Rodrigues(solvRR_inv)[0].squeeze(1)
            query_qwxyz = Quaternion(matrix=solvRR_inv).elements
            query_xyz = solvtt.squeeze(1)
            ans.append({
                "floor": q_dataset.get_path(q).split("/")[-5],
                "name" : q_dataset.get_path(q).split("/")[-1],
                "qw": query_qwxyz[0],
                "qx": query_qwxyz[1],
                "qy": query_qwxyz[2],
                "qz": query_qwxyz[3],
                "x": query_xyz[0],
                "y": query_xyz[1],
                "z": query_xyz[2],
            })
        except:
            ans.append({
                "floor": q_dataset.get_path(q).split("/")[-5],
                "name" : q_dataset.get_path(q).split("/")[-1],
                "qw": 0,
                "qx": 0,
                "qy": 0,
                "qz": 0,
                "x": 0,
                "y": 0,
                "z": 0
            })
            num_strange += 1
        del pair_3d, q_pairs
        if q % 100 == 0:
            with open("{}_superglue_rank_knn_num_{}_rerank_knn_num_{}_answer_{}_{}.json".format(args.globaldesc, args.rank_knn_num, args.rerank_knn_num, args.floor, q), 'w') as fp:
                json.dump(ans, fp)
        pbar.set_description("I dont know: {}".format(num_strange))
    return ans

def load_pretrained_layers(model, path) :
    
    state_dict = model.state_dict()
    param_names = list(state_dict.keys())  

    # load checkpoint
    pretrained_base_state_dict = torch.load(path)['state_dict']
    pretrained_base_state_dict_name = list(pretrained_base_state_dict.keys())

    # Transfer conv. parameters from pretrained model to current model
    for i, param in enumerate(param_names[:]):
        state_dict[param] = pretrained_base_state_dict[pretrained_base_state_dict_name[i]]

    model.load_state_dict(state_dict)

if __name__=='__main__':
    query_dataset = HyundaiTest(args.dataset_path, args.floor, "test", input_transform())
    db_dataset = Hyundai(args.dataset_path,
                         args.floor, "train", input_transform())
    query_dict = np.load(
        "{}_query_{}_features.npy".format(args.globaldesc, args.floor))
    db_dict = np.load("{}_db_{}_features.npy".format(args.globaldesc, args.floor))

    ind = ranking(query_dict, db_dict)
    neighbor, total_match = local_match(ind, query_dataset, db_dataset)
    # neighbor = np.load("netvlad_superglue_reranked_b1_db.npy")
    np.save("{}_superglue_reranked_{}_db.npy".format(args.globaldesc, args.floor), neighbor)
    # with open("netvlad_superglue_match_b1.pkl", "rb") as fp:
    #     total_match = pickle.load(fp)
    with open("{}_superglue_match_{}.pkl".format(args.globaldesc, args.floor), "wb") as fp:
        pickle.dump(total_match, fp)

    ans = pose_estimation(neighbor, total_match, query_dataset, db_dataset, (512, 512))
    with open("{}_superglue_rank_knn_num_{}_rerank_knn_num_{}_answer_{}.json".format(args.globaldesc, args.rank_knn_num, args.rerank_knn_num, args.floor), 'w') as fp:
        json.dump(ans, fp)
