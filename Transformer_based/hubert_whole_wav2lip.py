from os.path import dirname, join, basename, isfile
# 导入 `os.path` 模块中的特定函数，用于处理文件路径。
# `dirname` 返回文件的目录名。
# `join` 将多个路径组合成一个路径。
# `basename` 返回路径的基本名（文件名）。
# `isfile` 判断路径是否是文件。

from tqdm import tqdm
# `tqdm` 是一个进度条库，可以在循环中显示进度条，方便查看任务进度。

import cv2
# `cv2` 是 OpenCV 库的接口，用于计算机视觉任务，如图像处理、视频捕捉、物体检测等。
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data as data_utils
# `torch` 是一个流行的深度学习框架，提供了张量运算、自动求导、神经网络等功能。
# `torch.nn` 模块包含了构建神经网络的基本元素，如各种层（layers）、激活函数（activations）、损失函数（losses）等。
# `torch.optim` 模块提供了优化算法，如随机梯度下降（SGD）、Adam 等，用于更新模型参数。
# `torch.utils.data` 模块提供了数据加载器（DataLoader）和数据集（Dataset）接口，用于处理和加载训练数据。

import torchvision
# `torchvision` 是 PyTorch 的一个子库，专门用于计算机视觉任务。它提供了常用的数据集（如 ImageNet、CIFAR-10 等）、预训练模型和图像变换（transforms）工具。

import numpy as np
# `numpy` 是一个用于科学计算的库，支持多维数组和矩阵运算，以及大量的数学函数。

from glob import glob
# `glob` 模块用于查找符合特定模式的文件路径名，如查找特定扩展名的所有文件。

import os, random, argparse
# `os` 模块提供了与操作系统交互的功能，如文件和目录操作。
# `random` 模块用于生成随机数或进行随机操作。
# `argparse` 模块用于处理命令行参数，帮助解析用户输入的命令行选项和参数。

from hparams import hparams, get_image_list
# `hparams` 模块通常用于存储和管理超参数（hyperparameters），如学习率、批次大小等。`get_image_list` 可能是一个函数，用于获取图像文件列表。

import mediapipe as mp
# `mediapipe` 是一个跨平台的机器学习解决方案库，提供了计算机视觉、手势识别、人脸检测、姿态估计等功能的高效实现。

from VGG_LOSS import VGGLoss
# `VGGLoss` 可能是自定义的一个损失函数，基于 VGG 网络，用于计算感知损失（Perual Loss）或特征损失（Feature Loss）。

from models import Reconstructor_hubert_wav2lip
from models import SyncNet_both_hubert as SyncNet
# 这两行导入了自定义的模型模块。
# `Reconstructor_hubert_wav2lip` 可能是一个重建模型，用于处理音频或视频同步任务（如 Wav2Lip）。
# `SyncNet_both_hubert`（重命名为 `SyncNet`）可能是一个用于音频和视频同步的网络模型。

parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model without the visual quality discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)

parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)

parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', required=True,
                    type=str)
args = parser.parse_args()

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()

syncnet_T = 5
syncnet_mel_step_size = 10
syncnet_mel_step_size1 = 16

index_mouth = [0, 79, 43, 36]


class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)  # 开始帧序号
        vidname = dirname(start_frame)  # 返回除文件名外的路径

        img_window_fnames = []
        landmarks_window_fnames = []

        for frame_id in range(start_id, start_id + syncnet_T):
            img_frame = join(vidname, '{}.npy'.format(frame_id))
            landmark_frame = join(vidname, '{}_landmark.npy'.format(frame_id))
            if not isfile(img_frame):
                return None, None, None
            img_window_fnames.append(img_frame)
            landmarks_window_fnames.append(landmark_frame)

        landmarks_before_fnames = []
        for frame_id2 in range(start_id - 5, start_id):
            landmark_frame1 = join(vidname, '{}_landmark.npy'.format(frame_id2))
            if not isfile(landmark_frame1):
                return None, None, None
            landmarks_before_fnames.append(landmark_frame1)

        return img_window_fnames, landmarks_window_fnames, landmarks_before_fnames

    def read_window(self, window_fnames, landmarks_fnames, befor_landmarks_fnames):
        if window_fnames is None: return None, None
        if landmarks_fnames is None: return None, None

        landmarks_all = []
        for fname in landmarks_fnames:
            landmarks = np.load(fname)
            if len(landmarks) == 0:
                return None, None, None
            if landmarks is None:
                return None, None, None
            landmarks_all.append(landmarks)
        befor_landmarks_all = []
        for fname in befor_landmarks_fnames:
            before_landmarks = np.load(fname)
            if len(before_landmarks) == 0:
                return None, None, None
            if before_landmarks is None:
                return None, None, None
            befor_landmarks_all.append(before_landmarks)

        window = []
        for fname in window_fnames:
            img = np.load(fname)
            if len(img) == 0:
                return None, None, None
            if img is None:
                return None, None, None
            window.append(img)

        return window, landmarks_all, befor_landmarks_all

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)  # 0-indexing ---> 1-indexing
        start_idx = (int(start_frame_num)) * 2
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx: end_idx, :]

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*[0-9].npy')))

            if len(img_names) <= 3 * syncnet_T:
                continue

            img_name = random.choice(img_names)

            true_fnames, true_landmarks_fnames, befor_landmarks_fnames = self.get_window(img_name)

            if true_landmarks_fnames is None or len(true_landmarks_fnames) == 0:
                continue
            if befor_landmarks_fnames is None or len(befor_landmarks_fnames) == 0:
                continue
            if true_fnames is None or len(true_fnames) == 0:
                continue

            true_window, true_landmarks, befor_landmarks = self.read_window(true_fnames, true_landmarks_fnames, befor_landmarks_fnames)
            # 5, 96, 96, 3

            if true_window is None or len(true_window) == 0:
                continue
            if true_landmarks is None or len(true_landmarks) == 0:
                continue
            if befor_landmarks is None or len(befor_landmarks) == 0:
                continue

            window_copy = [row.copy() for row in true_window]

            for i_landmarks in range(5):
                x_index = [true_landmarks[i_landmarks][0][0], true_landmarks[i_landmarks][79][0],
                           true_landmarks[i_landmarks][43][0], true_landmarks[i_landmarks][36][0]]
                y_index = [true_landmarks[i_landmarks][0][1], true_landmarks[i_landmarks][79][1],
                           true_landmarks[i_landmarks][43][1], true_landmarks[i_landmarks][36][1]]
                y_min = min(y_index)
                y_max = max(y_index)
                x_min = min(x_index)
                x_max = max(x_index)
                if y_max + 2 >= 96:
                    y_max = 93
                if x_max + 2 >= 96:
                    x_max = 93
                for i_x in range(y_min - 2, y_max + 2):
                    for i_y in range(x_min - 2, x_max + 2):
                        true_window[i_landmarks][i_x][i_y] = 0.

            window = self.prepare_window(true_window)
            window_gt = self.prepare_window(window_copy)

            try:
                wavpath = join(vidname, "audiohubert.npy")
                orig_mel = np.load(wavpath)


            except Exception as e:
                continue
            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if mel.shape[0] != syncnet_mel_step_size:
                continue


            g = torch.FloatTensor(window)
            gt = torch.FloatTensor(window_gt)
            true_landmarks = torch.FloatTensor(np.array(true_landmarks) / 96.)
            befor_landmarks = torch.FloatTensor(np.array(befor_landmarks) / 96.)
            mel = torch.FloatTensor(mel)

            return g, gt, true_landmarks, befor_landmarks, mel


def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((x, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])


device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False

recon_loss = nn.L1Loss()
mse_loss = nn.MSELoss()
criterion = nn.MSELoss()
logloss = nn.BCELoss()
perceptual_loss = VGGLoss()

def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def get_sync_loss(mel, g, landmarks):
    g = g[:, :, :, g.size(3) // 2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v, l = syncnet(mel, g, landmarks)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v + l, y)

index = [61, 76, 185, 146, 62, 184, 183, 78, 77, 95, 96, 191, 91, 90, 89, 88, 80, 42, 74, 40, 39, 73, 41, 81, 178, 179,
         180, 181, 37, 72, 38, 82, 87, 86, 85, 84, 0, 11, 12, 13, 14, 15, 16, 17, 267, 302, 268, 312, 317, 316, 315,
         314, 405, 404, 403, 402, 311, 271, 303, 269, 270, 304, 272, 310, 318, 319, 320, 321, 409, 408, 407, 415, 324,
         325, 307, 375, 308, 292, 306, 291, 57, 287]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

w_p = 0.01
w_landmark = 0.01



def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch, w_p, w_landmark
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_p_loss, running_l1_loss, running_landmark_loss, running_sync_loss = 0., 0., 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, gt, true_landmarks, befor_landmarks, mel) in prog_bar:
            model.train()
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device)
            mel = mel.to(device)

            gt = gt.to(device)
            true_landmarks = true_landmarks.to(device)
            befor_landmarks = befor_landmarks.to(device)


            g, predicted_landmarks = model(mel, befor_landmarks, x)  # B, 3, 5, 96, 96

            # ******---------------------------calculate whole face perceptual loss!---------------------******#
            g_p = g.transpose(1, 2)
            g_p = torch.cat([g_p[i, :, :, :] for i in range(g_p.size(0))], dim=0)
            gt_p = gt.transpose(1, 2)
            gt_p = torch.cat([gt_p[i, :, :, :] for i in range(gt_p.size(0))], dim=0)
            # ******---------------------------calculate whole face perceptual loss!---------------------******#
            p_loss = perceptual_loss(g_p, gt_p)
            l1loss = recon_loss(g, gt)
            landmark_loss = criterion(true_landmarks, predicted_landmarks)
            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g, predicted_landmarks)
            else:
                sync_loss = 0.

            loss = (1 - hparams.syncnet_wt - w_p - w_landmark)*l1loss + w_p*p_loss + w_landmark*landmark_loss + hparams.syncnet_wt*sync_loss
            loss.backward()
            optimizer.step()

            if global_step % checkpoint_interval == 0:
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

            global_step += 1

            running_l1_loss += l1loss.item()
            running_p_loss += p_loss.item()
            running_landmark_loss += landmark_loss.item()
            if hparams.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step == 1 or global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    average_sync_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir)
                    if average_sync_loss < 0.65:
                        hparams.set_hparam('syncnet_wt', 0.01)
                        w_p = 0.
                        w_landmark = 0.


            prog_bar.set_description(
                'Train L1: {}, Perceptual: {}, Landmark: {}, Sync: {}, w_p:{}, w_landmark:{}'.format(running_l1_loss / (step + 1), running_p_loss / (step + 1), running_landmark_loss / (step+1), running_sync_loss / (step +1), w_p, w_landmark))

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 300
    print('Evaluating for {} steps'.format(eval_steps))
    recon_losses, perceptual_losses, landmark_losses, sync_losses = [], [], [], []
    step = 0
    while 1:
        for x, gt, true_landmarks, befor_landmarks, mel in test_data_loader:
            step += 1
            model.eval()

            # Move data to CUDA device
            x = x.to(device)
            gt = gt.to(device)
            mel = mel.to(device)

            true_landmarks = true_landmarks.to(device)
            befor_landmarks = befor_landmarks.to(device)


            g, predicted_landmarks = model(mel, befor_landmarks, x)

            landmark_loss = criterion(true_landmarks, predicted_landmarks)
            sync_loss = get_sync_loss(mel, g, predicted_landmarks)
            g_p = g.transpose(1, 2)
            g_p = torch.cat([g_p[i, :, :, :] for i in range(g_p.size(0))], dim=0)
            gt_p = gt.transpose(1, 2)
            gt_p = torch.cat([gt_p[i, :, :, :] for i in range(gt_p.size(0))], dim=0)

            p_loss = perceptual_loss(g_p, gt_p)
            l1loss = recon_loss(g, gt)

            recon_losses.append(l1loss.item())
            perceptual_losses.append(p_loss.item())
            landmark_losses.append(landmark_loss)
            sync_losses.append(sync_loss)

            if step > eval_steps:
                averaged_recon_loss = sum(recon_losses) / len(recon_losses)
                averaged_p_loss = sum(perceptual_losses) / len(perceptual_losses)
                averaged_landmark_loss = sum(landmark_losses) / len(landmark_losses)
                averaged_sync_loss = sum(sync_losses) / len(sync_losses)
                print('Eval L1: {}, Perceptual: {}, Landmark: {}, Sync: {}'.format(averaged_recon_loss, averaged_p_loss,averaged_landmark_loss, averaged_sync_loss))
                return averaged_sync_loss


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model


if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir

    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)

    test_data_loader = data_utils.DataLoader(test_dataset, batch_size=16, num_workers=4)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = Reconstructor_hubert_wav2lip().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate)

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)

    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Train!
    train(device, model, train_data_loader, test_data_loader, optimizer, checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.checkpoint_interval, nepochs=hparams.nepochs)
