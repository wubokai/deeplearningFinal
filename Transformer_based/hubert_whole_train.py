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

from models import Hubert_reconstructor
from models import SyncNet_both_hubert as SyncNet

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

index_mouth = [0, 79, 43, 36]


class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame, wrong_start_frame):
        start_id = self.get_frame_id(start_frame)  # 开始帧序号
        vidname = dirname(start_frame)  # 返回除文件名外的路径
        wrong_start_id = self.get_frame_id(wrong_start_frame)  # 开始帧序号
        wrong_vidname = dirname(wrong_start_frame)  # 返回除文件名外的路径

        img_window_fnames = []
        landmarks_window_fnames = []

        for frame_id in range(start_id, start_id + syncnet_T):
            img_frame = join(vidname, '{}.npy'.format(frame_id))
            landmark_frame = join(vidname, '{}_landmark.npy'.format(frame_id))
            if not isfile(img_frame):
                return None, None, None, None
            img_window_fnames.append(img_frame)
            landmarks_window_fnames.append(landmark_frame)

        landmarks_before_fnames = []
        for frame_id2 in range(start_id - 5, start_id):
            landmark_frame1 = join(wrong_vidname, '{}_landmark.npy'.format(frame_id2))
            if not isfile(landmark_frame1):
                return None, None, None, None
            landmarks_before_fnames.append(landmark_frame1)

        wrong_window_fnames = []
        for wrong_frame_id in range(wrong_start_id, wrong_start_id + syncnet_T):
            wrong_img_frame = join(vidname, '{}.npy'.format(wrong_frame_id))

            if not isfile(wrong_img_frame):
                return None, None, None, None
            wrong_window_fnames.append(wrong_img_frame)


        return img_window_fnames, landmarks_window_fnames, landmarks_before_fnames, wrong_window_fnames

    def read_window(self, window_fnames, landmarks_fnames, befor_landmarks_fnames, wrong_window_fnames):
        if window_fnames is None: return None, None, None
        if landmarks_fnames is None: return None, None, None

        landmarks_all = []
        for fname1 in landmarks_fnames:
            landmarks = np.load(fname1)
            if len(landmarks) == 0:
                return None, None, None, None
            if landmarks is None:
                return None, None, None, None
            landmarks_all.append(landmarks)
        befor_landmarks_all = []
        for fname2 in befor_landmarks_fnames:
            before_landmarks = np.load(fname2)
            if len(before_landmarks) == 0:
                return None, None, None, None
            if before_landmarks is None:
                return None, None, None, None
            befor_landmarks_all.append(before_landmarks)

        window = []
        for fname3 in window_fnames:
            img3 = np.load(fname3)
            if len(img3) == 0:
                return None, None, None, None
            if img3 is None:
                return None, None, None, None
            window.append(img3)

        wrong_window = []
        for fname4 in wrong_window_fnames:
            img4 = np.load(fname4)
            if len(img4) == 0:
                return None, None, None, None
            if img4 is None:
                return None, None, None, None
            wrong_window.append(img4)
        return window, landmarks_all, befor_landmarks_all, wrong_window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)  # 0-indexing ---> 1-indexing
        start_idx = (int(start_frame_num)) * 2
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx: end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1  # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0:
            return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m)
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
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            true_fnames, true_landmarks_fnames, befor_landmarks_fnames, wrong_window_fnames = self.get_window(img_name, wrong_img_name)

            if true_landmarks_fnames is None:
                continue
            if befor_landmarks_fnames is None:
                continue
            if true_fnames is None:
                continue
            if wrong_window_fnames is None:
                continue

            true_window, true_landmarks, befor_landmarks, wrong_window = self.read_window(true_fnames, true_landmarks_fnames, befor_landmarks_fnames, wrong_window_fnames)
            # 5, 96, 96, 3

            if true_window is None or len(true_window) == 0:
                continue
            if true_landmarks is None or len(true_landmarks) == 0:
                continue
            if befor_landmarks is None or len(befor_landmarks) == 0:
                continue
            if wrong_window is None or len(wrong_window) == 0:
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
                if y_max + 6 >= 96:
                    y_max = 89
                if x_max + 6 >= 96:
                    x_max = 89
                for i_x in range(y_min - 6, y_max + 6):
                    for i_y in range(x_min - 6, x_max + 6):
                        true_window[i_landmarks][i_x][i_y] = 0.

            window = self.prepare_window(true_window)
            window_gt = self.prepare_window(window_copy)
            window_reference = self.prepare_window(wrong_window)
            x = np.concatenate([window, window_reference], axis=0)

            try:
                wavpath = join(vidname, "audiohubert.npy")
                orig_mel = np.load(wavpath)
            except Exception as e:
                continue
            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue
            if mel.shape[0] != syncnet_mel_step_size:
                continue

            g = torch.FloatTensor(x)
            gt = torch.FloatTensor(window_gt)
            true_landmarks = torch.FloatTensor(np.array(true_landmarks) / 96.)
            befor_landmarks = torch.FloatTensor(np.array(befor_landmarks) / 96.)
            mel = torch.FloatTensor(mel)
            indiv_mels = torch.FloatTensor(indiv_mels)

            return g, gt, true_landmarks, befor_landmarks, mel, indiv_mels


def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
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

w_landmark = 0.01
def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch, w_landmark
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_l1_loss, running_landmark_loss, running_sync_loss = 0., 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, gt, true_landmarks, befor_landmarks, mel, indiv_mels) in prog_bar:
            model.train()
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)
            true_landmarks = true_landmarks.to(device)
            befor_landmarks = befor_landmarks.to(device)

            g, predicted_landmarks = model(indiv_mels, befor_landmarks, x)  # B, 3, 5, 96, 96

            l1loss = recon_loss(g, gt)
            landmark_loss = criterion(true_landmarks, predicted_landmarks)
            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g, predicted_landmarks)
            else:
                sync_loss = 0.

            loss = (1 - hparams.syncnet_wt - w_landmark)*l1loss + w_landmark*landmark_loss + hparams.syncnet_wt*sync_loss
            loss.backward()
            optimizer.step()

            if global_step % checkpoint_interval == 0:
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

            global_step += 1

            running_l1_loss += l1loss.item()
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
                        hparams.set_hparam('syncnet_wt', 0.02)

            prog_bar.set_description(
                'Train L1: {}, Landmark: {}, Sync: {}'.format(running_l1_loss / (step + 1), running_landmark_loss / (step+1), running_sync_loss / (step +1)))

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 300
    print('Evaluating for {} steps'.format(eval_steps))
    recon_losses, landmark_losses, sync_losses = [], [], []
    step = 0
    while 1:
        for x, gt, true_landmarks, befor_landmarks, mel, indiv_mels in test_data_loader:
            step += 1
            model.eval()

            # Move data to CUDA device
            x = x.to(device)
            gt = gt.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            true_landmarks = true_landmarks.to(device)
            befor_landmarks = befor_landmarks.to(device)

            g, predicted_landmarks = model(indiv_mels, befor_landmarks, x)

            landmark_loss = criterion(true_landmarks, predicted_landmarks)
            sync_loss = get_sync_loss(mel, g, predicted_landmarks)
            l1loss = recon_loss(g, gt)

            recon_losses.append(l1loss.item())
            landmark_losses.append(landmark_loss)
            sync_losses.append(sync_loss)

            if step > eval_steps:
                averaged_recon_loss = sum(recon_losses) / len(recon_losses)
                averaged_landmark_loss = sum(landmark_losses) / len(landmark_losses)
                averaged_sync_loss = sum(sync_losses) / len(sync_losses)
                print('Eval L1: {}, Landmark: {}, Sync: {}'.format(averaged_recon_loss, averaged_landmark_loss, averaged_sync_loss))
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
    train_data_loader = data_utils.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
    test_data_loader = data_utils.DataLoader(test_dataset, batch_size=16, num_workers=4)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = Hubert_reconstructor().to(device)
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
