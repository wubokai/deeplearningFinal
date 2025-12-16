from hparams import get_image_list
# 从 `hparams` 模块导入 `get_image_list`，这是一个自定义函数，可能用于获取图片文件列表，帮助加载或处理图像数据。

from os.path import basename, dirname, join, isfile
# 从 `os.path` 模块导入特定的路径操作函数：
# - `basename` 返回文件的基本名称（文件名）。
# - `dirname` 返回文件所在的目录。
# - `join` 将多个路径片段组合成一个完整的路径。
# - `isfile` 判断路径是否指向一个有效的文件。

import random, argparse
# `random` 模块用于生成随机数或进行随机选择操作。
# `argparse` 模块用于解析命令行参数，方便用户通过命令行提供输入和选项。

from hparams import hparams
# `hparams` 是自定义模块，通常用于存储程序运行所需的超参数（hyperparameters），如模型训练中的学习率、批次大小等。

from glob import glob
# `glob` 模块用于查找符合特定模式的文件路径名，例如查找某个文件夹下所有符合条件的文件。

import numpy as np
# `numpy` 是一个用于科学计算的库，提供了多维数组支持以及高效的数值计算函数，广泛用于数据处理和机器学习中。

import torch
# `torch` 是一个流行的深度学习框架，支持高效的张量运算和自动求导，用于构建和训练深度学习模型。

from torch import nn
# `torch.nn` 是 PyTorch 中的模块，提供了构建神经网络的基本组件，例如各种神经网络层（如卷积层、全连接层）、激活函数、损失函数等。

from torch import optim
# `torch.optim` 提供了各种优化算法，如随机梯度下降（SGD）、Adam 等，用于调整神经网络的参数以最小化损失函数。

from torch.utils import data as data_utils
# `torch.utils.data` 模块提供了数据集（Dataset）和数据加载器（DataLoader）的接口，用于高效地处理和加载训练数据。

import tqdm
# `tqdm` 是一个用于显示进度条的库，常用于循环操作中显示当前任务的进度，提升代码可读性和用户体验。

import os
# `os` 模块提供了与操作系统交互的功能，如文件、目录操作、环境变量管理等。

from models import hubert_former
# 从 `models` 模块中导入 `hubert_former`，这是自定义的模型，可能基于 HuB（Hidden Unit BERT）架构，用于音频相关任务，如语音识别或语音增强。


parser = argparse.ArgumentParser(description='Code to train the Landmark_former model!!!!')
parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
args = parser.parse_args()

syncnet_T = 5
syncnet_mel_step_size = 10

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))


class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)  # 开始帧序号
        vidname = dirname(start_frame)  # 返回除文件名外的路径

        landmarks_window_fnames = []
        for frame_id1 in range(start_id, start_id + 5):
            landmark_frame1 = join(vidname, '{}_landmark.npy'.format(frame_id1))
            if not isfile(landmark_frame1):
                return None, None
            landmarks_window_fnames.append(landmark_frame1)

        landmarks_before_fnames = []
        for frame_id2 in range(start_id - 5, start_id):
            landmark_frame1 = join(vidname, '{}_landmark.npy'.format(frame_id2))
            if not isfile(landmark_frame1):
                return None, None
            landmarks_before_fnames.append(landmark_frame1)

        return landmarks_window_fnames, landmarks_before_fnames

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

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)  # 随机得到一个范围内的整数
            vidname = self.all_videos[idx]  # 找到这个文件夹
            img_names = list(glob(join(vidname, '*[0-9].npy')))  # 获取视频帧

            if len(img_names) <= 3 * syncnet_T:  # 如果帧数不足15个，跳过这条视频数据
                continue
            img_name = random.choice(img_names)  # 随机选定一条帧
            chosen = img_name

            window, before_window = self.get_window(chosen)  # 获取选择的数据
            if window is None:
                continue

            all_read = True

            True_landmarks = []
            for True_frame in window:
                true_frame = np.load(True_frame)
                if len(true_frame) == 0 or true_frame is None:
                    all_read = False
                    break
                True_landmarks.append(true_frame)

            berfore_landmarks = []
            for before_frame in before_window:
                Before_frame = np.load(before_frame)
                if len(Before_frame) == 0 or Before_frame is None:
                    all_read = False
                    break
                berfore_landmarks.append(Before_frame)

            if not all_read: continue

            try:
                wavpath = join(vidname, "audiohubert.npy")
                orig_mel = np.load(wavpath)  # 获取音频对应的mfcc

            except Exception as e:
                continue

            mel = self.get_segmented_mels(orig_mel.copy(), img_name)
            if mel is None:
                continue


            mel = torch.FloatTensor(mel)
            gt = torch.tensor(np.array(True_landmarks), dtype=torch.float32)/96.
            gb = torch.tensor(np.array(berfore_landmarks), dtype=torch.float32)/96.

            return mel, gt, gb


mse_loss = nn.MSELoss(reduction='mean')
recon_loss = nn.L1Loss()


def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step
    while global_epoch < nepochs:
        running_loss = 0.
        prog_bar = tqdm.tqdm(enumerate(train_data_loader))
        for step, (mel, landmarks, before_landmarks) in prog_bar:
            model.train()
            optimizer.zero_grad()

            mel = mel.to(device)
            landmarks = landmarks.to(device)
            before_landmarks = before_landmarks.to(device)
            pt = model(mel, before_landmarks)

            loss = recon_loss(pt, landmarks)
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)
            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

            prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))
            global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        print("start eval!!!")
        for step, (mel, landmarks, before_landmarks) in enumerate(test_data_loader):

            model.eval()

            mel = mel.to(device)
            # Transform data to CUDA device
            landmarks = landmarks.to(device)
            before_landmarks = before_landmarks.to(device)

            pt = model(mel, before_landmarks)

            loss = recon_loss(landmarks, pt)
            losses.append(loss.item())

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)
        print("evaling!!!")

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


def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model


if __name__ == '__main__':
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = Dataset("train")
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4)

    test_dataset = Dataset('val')
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=16, shuffle=True, num_workers=4)

    # Model
    model = hubert_former().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=0.00001)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)
