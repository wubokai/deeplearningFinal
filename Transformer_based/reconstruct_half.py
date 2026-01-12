from os.path import dirname, join, basename, isfile
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list
import mediapipe as mp

from models import Reconstructor

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

            true_fnames, true_landmarks_fnames = self.get_window(img_name)
            wrong_fnames, wrong_landmarks_fnames = self.get_window(wrong_img_name)

            if true_fnames is None or wrong_fnames is None:
                continue

            if true_landmarks_fnames is None or len(true_landmarks_fnames) is None:
                continue

            true_window, true_landmarks = self.read_window(true_fnames, true_landmarks_fnames)

            if true_window is None or len(true_window) == 0:
                continue
            if true_landmarks is None or len(true_landmarks) == 0:
                continue

            wrong_window, wrong_landmarks = self.read_window(wrong_fnames, wrong_landmarks_fnames)

            if wrong_window is None or len(wrong_window) == 0:
                continue
            if wrong_landmarks is None or len(wrong_landmarks) == 0:
                continue

            window = self.prepare_window(true_window)
            y = window.copy()
            window[:, :, window.shape[2] // 2:] = 0.  #

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            y = torch.FloatTensor(y)
            landmarks = torch.FloatTensor(np.array(true_landmarks)/82)


            return x, y, landmarks


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


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


device = torch.device("cuda" if use_cuda else "cpu")
recon_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

def get_sync_loss(g, gt):
    loss = mse_loss(g, gt)
    return loss


index = [61, 76, 185, 146, 62, 184, 183, 78, 77, 95, 96, 191, 91, 90, 89, 88, 80, 42, 74, 40, 39, 73, 41, 81, 178, 179,
         180, 181, 37, 72, 38, 82, 87, 86, 85, 84, 0, 11, 12, 13, 14, 15, 16, 17, 267, 302, 268, 312, 317, 316, 315,
         314, 405, 404, 403, 402, 311, 271, 303, 269, 270, 304, 272, 310, 318, 319, 320, 321, 409, 408, 407, 415, 324,
         325, 307, 375, 308, 292, 306, 291, 57, 287]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()


def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_l1_loss = 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, gt, landmarks) in prog_bar:
            model.train()
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device)

            gt = gt.to(device)
            landmarks = landmarks.to(device)

            g = model(landmarks, x)  # B, 3, 5, 96, 96

            g_landmark = g.permute(0, 2, 3, 4, 1).detach()
            batch_landmarks = []
            all_yes = True
            for i in range(g_landmark.shape[0]):
                step_landmarks = []
                for j in range(g_landmark.shape[1]):
                    single_landmarks = []
                    img_landmark = np.asarray((g_landmark[i][j].cpu()) * 255)
                    gray = cv2.cvtColor(img_landmark.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    height, width, _ = img_landmark.shape
                    results = face_mesh.process(gray)
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            for k in index:
                                x_1, y_1 = int(face_landmarks.landmark[k].x * width), int(
                                    face_landmarks.landmark[k].y * height)
                                single_landmarks.append([x_1, y_1])
                    if len(single_landmarks) == 0 or single_landmarks is None:
                        all_yes = False
                        break
                    step_landmarks.append(single_landmarks)
                if not all_yes:
                    break
                batch_landmarks.append(step_landmarks)
            if all_yes:
                batch_landmarks = torch.tensor(np.array(batch_landmarks)/82)
                batch_landmarks = batch_landmarks.to(device)
                batch_landmarks = batch_landmarks.float()
                sync_loss = get_sync_loss(landmarks, batch_landmarks)

            else:
                sync_loss = torch.tensor(0.99)

            l1loss = recon_loss(g, gt)

            loss = 0.97 * l1loss + 0.03 * sync_loss
            loss.backward()
            optimizer.step()

            if global_step % checkpoint_interval == 0:
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

            global_step += 1

            running_l1_loss += l1loss.item()
            running_sync_loss += sync_loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step == 1 or global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir)


            prog_bar.set_description('Train L1: {} ,Sync Loss:{}'.format(running_l1_loss / (step + 1), running_sync_loss / (step + 1)))

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 300
    print('Evaluating for {} steps'.format(eval_steps))
    recon_losses, sync_losses = [], []
    step = 0
    while 1:
        for x, gt, landmarks in test_data_loader:
            step += 1
            model.eval()

            # Move data to CUDA device
            x = x.to(device)
            gt = gt.to(device)
            landmarks = landmarks.to(device)

            g = model(landmarks, x)
            g_landmark = g.permute(0, 2, 3, 4, 1).detach()
            batch_landmarks = []
            all_yes = True
            for i in range(g_landmark.shape[0]):
                step_landmarks = []
                for j in range(g_landmark.shape[1]):
                    single_landmarks = []
                    img_landmark = np.asarray((g_landmark[i][j].cpu()) * 255)
                    gray = cv2.cvtColor(img_landmark.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    height, width, _ = img_landmark.shape
                    results = face_mesh.process(gray)
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            for k in index:
                                x_1, y_1 = int(face_landmarks.landmark[k].x * width), int(
                                    face_landmarks.landmark[k].y * height)
                                single_landmarks.append([x_1, y_1])
                    if len(single_landmarks) == 0 or single_landmarks is None:
                        all_yes = False
                        break
                    step_landmarks.append(single_landmarks)
                if not all_yes:
                    break
                batch_landmarks.append(step_landmarks)
            if all_yes:
                batch_landmarks = torch.tensor(np.array(batch_landmarks) / 82)
                batch_landmarks = batch_landmarks.to(device)
                batch_landmarks = batch_landmarks.float()
                sync_loss = get_sync_loss(landmarks, batch_landmarks)

            else:
                sync_loss = torch.tensor(0.99)

            l1loss = recon_loss(g, gt)
            recon_losses.append(l1loss.item())
            sync_losses.append(sync_loss.item())

            if step > eval_steps:
                averaged_recon_loss = sum(recon_losses) / len(recon_losses)
                averaged_sync_loss = sum(sync_losses) / len(sync_losses)
                print('Eval L1: {}, Sync Loss: {}'.format(averaged_recon_loss, averaged_sync_loss))
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
    model = Reconstructor().to(device)
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
