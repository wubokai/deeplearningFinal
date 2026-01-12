from os.path import dirname, join, basename, isfile
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data as data_utils
import torchvision
import numpy as np

from glob import glob

import os, random, argparse
from hparams import hparams, get_image_list
import mediapipe as mp

from VGG_LOSS import VGGLoss
from models import Reconstructor_lip_audio


parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model without the visual quality discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)

parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)

args = parser.parse_args()

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()

syncnet_T = 5
syncnet_mel_step_size = 16

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
                return None, None
            img_window_fnames.append(img_frame)
            landmarks_window_fnames.append(landmark_frame)
        return img_window_fnames, landmarks_window_fnames

    def read_window(self, window_fnames, landmarks_fnames):
        if window_fnames is None: return None, None
        if landmarks_fnames is None: return None, None

        landmarks_all = []
        for fname in landmarks_fnames:
            landmarks = np.load(fname)
            if len(landmarks) == 0:
                return None, None
            if landmarks is None:
                return None, None
            landmarks_all.append(landmarks)

        window = []
        for fname in window_fnames:
            img = np.load(fname)
            if len(img) == 0:
                return None, None
            if img is None:
                return None, None
            window.append(img)

        return window, landmarks_all

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)  # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx: end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1  # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

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

            true_fnames, true_landmarks_fnames = self.get_window(img_name)

            if true_landmarks_fnames is None or len(true_landmarks_fnames) is None:
                continue

            true_window, true_landmarks = self.read_window(true_fnames, true_landmarks_fnames)
            # 5, 96, 96, 3

            if true_window is None or len(true_window) == 0:
                continue
            if true_landmarks is None or len(true_landmarks) == 0:
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
                wavpath = join(vidname, "audio.npy")
                orig_mel = np.load(wavpath)

            except Exception as e:
                continue
            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            if (mel.shape[0] != syncnet_mel_step_size):
                continue
            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            g = torch.FloatTensor(window)
            gt = torch.FloatTensor(window_gt)
            landmarks = torch.FloatTensor(np.array(true_landmarks) / 96.)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)

            return g, gt, landmarks, mel, indiv_mels


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


logloss = nn.BCELoss()
perceptual_loss = VGGLoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


device = torch.device("cuda" if use_cuda else "cpu")
recon_loss = nn.L1Loss()
mse_loss = nn.MSELoss()



index = [61, 76, 185, 146, 62, 184, 183, 78, 77, 95, 96, 191, 91, 90, 89, 88, 80, 42, 74, 40, 39, 73, 41, 81, 178, 179,
         180, 181, 37, 72, 38, 82, 87, 86, 85, 84, 0, 11, 12, 13, 14, 15, 16, 17, 267, 302, 268, 312, 317, 316, 315,
         314, 405, 404, 403, 402, 311, 271, 303, 269, 270, 304, 272, 310, 318, 319, 320, 321, 409, 408, 407, 415, 324,
         325, 307, 375, 308, 292, 306, 291, 57, 287]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
w_l1 = 0.9
w_p = 0.1


def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_p_loss, running_l1_loss = 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, gt, landmarks, mel, indiv_mels) in prog_bar:
            model.train()
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)

            gt = gt.to(device)
            landmarks = landmarks.to(device)

            g = model(landmarks, x, indiv_mels)  # B, 3, 5, 96, 96

            # ******---------------------------calculate lip zone perceptual loss!---------------------******#
            # g_lip = torch.permute(g, (0, 2, 3, 4, 1))
            # gt_lip = torch.permute(gt, (0, 2, 3, 4, 1))
            # g_lip = torch.cat([g_lip[i, :, :, :] for i in range(g_lip.size(0))], dim=0)
            # gt_lip = torch.cat([gt_lip[i, :, :, :] for i in range(gt_lip.size(0))], dim=0)
            # landmarks_lip = torch.cat([landmarks[i, :, :, :] for i in range(landmarks.size(0))], dim=0)
            # g_lip_batch = []
            # gt_lip_batch = []
            # B = g_lip.shape[0]
            # for b in range(B):
            #     x_index = [landmarks_lip[b][0][0], landmarks_lip[b][79][0],
            #                landmarks_lip[b][43][0], landmarks_lip[b][36][0]]
            #     y_index = [landmarks_lip[b][0][1], landmarks_lip[b][79][1],
            #                landmarks_lip[b][43][1], landmarks_lip[b][36][1]]
            #     y_min = int(min(y_index) * 96)
            #     y_max = int(max(y_index) * 96)
            #     x_min = int(min(x_index) * 96)
            #     x_max = int(max(x_index) * 96)
            #     if y_max + 2 > 96:
            #         y_max = 93
            #     if x_max + 2 > 96:
            #         x_max = 93
            #     g_lip_step = []
            #     gt_lip_step = []
            #     for i_x in range(y_min - 2, y_max + 2):
            #         g_lip_k = []
            #         gt_lip_k = []
            #         for i_y in range(x_min - 2, x_max + 2):
            #             g_lip_k.append(g_lip[b][i_x][i_y])
            #             gt_lip_k.append(gt_lip[b][i_x][i_y])
            #         g_lip_step.append(torch.stack(g_lip_k))
            #         gt_lip_step.append(torch.stack(gt_lip_k))
            #     g_lip_batch.append(torch.stack(g_lip_step))
            #     gt_lip_batch.append(torch.stack(gt_lip_step))
            # p_lip_loss_sum = []
            # for lip in range(B):
            #     p_lip_loss = perceptual_loss((gt_lip_batch[lip].transpose(0, 2)).unsqueeze(0), (g_lip_batch[lip].transpose(0, 2)).unsqueeze(0))
            #     p_lip_loss_sum.append(p_lip_loss)
            #p_loss = sum(p_lip_loss_sum) / len(p_lip_loss_sum)
            # ******---------------------------calculate lip zone perceptual loss!---------------------******#


            # ******---------------------------calculate whole face perceptual loss!---------------------******#
            g_p = g.transpose(1, 2)
            g_p = torch.cat([g_p[i, :, :, :] for i in range(g_p.size(0))], dim=0)
            gt_p = gt.transpose(1, 2)
            gt_p = torch.cat([gt_p[i, :, :, :] for i in range(gt_p.size(0))], dim=0)
            p_loss = perceptual_loss(g_p, gt_p)
            # ******---------------------------calculate whole face perceptual loss!---------------------******#

            l1loss = recon_loss(g, gt)

            loss = w_l1 * l1loss + w_p * p_loss
            loss.backward()
            optimizer.step()

            if global_step % checkpoint_interval == 0:
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

            global_step += 1

            running_l1_loss += l1loss.item()
            running_p_loss += p_loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step == 1 or global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

            prog_bar.set_description(
                'Train L1: {}, Perceptual: {} :'.format(running_l1_loss / (step + 1), running_p_loss / (step + 1)))

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 300
    print('Evaluating for {} steps'.format(eval_steps))
    recon_losses, perceptual_losses = [], []
    step = 0
    while 1:
        for x, gt, landmarks, mel, indiv_mels in test_data_loader:
            step += 1
            model.eval()

            # Move data to CUDA device
            x = x.to(device)
            gt = gt.to(device)
            landmarks = landmarks.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)

            g = model(landmarks, x, indiv_mels)
            g_p = g.transpose(1, 2)

            g_p = torch.cat([g_p[i, :, :, :] for i in range(g_p.size(0))], dim=0)
            gt_p = gt.transpose(1, 2)
            gt_p = torch.cat([gt_p[i, :, :, :] for i in range(gt_p.size(0))], dim=0)

            p_loss = perceptual_loss(g_p, gt_p)
            l1loss = recon_loss(g, gt)

            recon_losses.append(l1loss.item())
            perceptual_losses.append(p_loss.item())

            if step > eval_steps:
                averaged_recon_loss = sum(recon_losses) / len(recon_losses)
                averaged_p_loss = sum(perceptual_losses) / len(perceptual_losses)
                print('Eval L1: {}, Perceptual: {}:'.format(averaged_recon_loss, averaged_p_loss))
                return


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
    model = Reconstructor_lip_audio().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate)

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Train!
    train(device, model, train_data_loader, test_data_loader, optimizer, checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.checkpoint_interval, nepochs=hparams.nepochs)
