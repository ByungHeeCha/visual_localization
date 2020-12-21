import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision import transforms, models
from data import Hyundai, HyundaiTest
from models import NetVLAD
from PIL import Image
import dirtorch.nets as nets

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--floor', type=str, help='floor: b1, 1f')
parser.add_argument('--globaldesc', type=str,
                    help='Global descriptor: netvlad, apgem')
parser.add_argument('--batchsize', type=int,
                    help='batch size')
parser.add_argument('--dataset_path', type=str, help='Path of a dataset', default="../dataset")
parser.add_argument('--checkpoint_path', type=str,
                    help='Path of a model checkpoint')

args = parser.parse_args()

def resize(img):
    w, h = img.size
    return transforms.Resize((h//2, w//2))(img)

def input_transform():
    return transforms.Compose([
        transforms.Resize((512, 512)),
        # transforms.Lambda(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def extract_feature_netvlad(dataloader, model):
    pbar = tqdm(dataloader)
    model.cuda()
    encodings = []
    for idx, image in enumerate(pbar):
        model.eval()
        img = image.cuda()
        with torch.no_grad():
            vlad_encoding = model.pool(model.encoder(img))
        encodings.append(vlad_encoding.cpu())
    enc_mat = torch.cat(encodings).numpy()
    return enc_mat
    
def extract_feature_apgem(dataloader, model):
    pbar = tqdm(dataloader)
    model.cuda()
    encodings = []
    for idx, image in enumerate(pbar):
        model.eval()
        img = image.cuda()
        with torch.no_grad():
            model_encoding = model(img)
        if len(model_encoding.shape) == 1:
            model_encoding = model_encoding.unsqueeze(0)
        encodings.append(model_encoding.cpu())
    enc_mat = torch.cat(encodings).numpy()
    return enc_mat

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
    dataset = Hyundai(args.dataset_path, args.floor, "train", input_transform())
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False)
    q_dataset = HyundaiTest(args.dataset_path, args.floor, "test", input_transform())
    q_dataloader = DataLoader(q_dataset, batch_size=args.batchsize, shuffle=False)
    if args.globaldesc == 'netvlad':
        encoder = models.vgg16(pretrained=True)
        layers = list(encoder.features.children())[:-2]
        encoder = nn.Sequential(*layers)
        model = nn.Module() 
        model.add_module('encoder', encoder)
        netvlad = NetVLAD(dim=512)
        model.add_module('pool', netvlad)
        load_pretrained_layers(model, args.checkpoint_path)
        enc_mat = extract_feature_netvlad(dataloader, model)
        enc_mat_q = extract_feature_netvlad(q_dataloader, model)
    elif args.globaldesc == 'apgem':
        model = nets.create_model(
            "resnet101_rmac", pretrained=args.checkpoint_path, without_fc=False)
        enc_mat = extract_feature_apgem(dataloader, model)
        enc_mat_q = extract_feature_apgem(q_dataloader, model)
    else:
        print("wrong args!")
        exit()
    np.save("{}_db_{}_features.npy".format(args.globaldesc, args.floor), enc_mat)
    np.save("{}_query_{}_features.npy".format(args.globaldesc, args.floor), enc_mat)
