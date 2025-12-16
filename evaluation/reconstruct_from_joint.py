from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess
from tqdm import tqdm
from glob import glob
import torch

sys.path.append('../')
import audio
import face_detection
from models import Hubert_reconstructor
import mediapipe as mp

parser = argparse.ArgumentParser(description='Code to generate results for test filelists')

parser.add_argument('--filelist', type=str,
                    help='Filepath of filelist file to read', required=True)
parser.add_argument('--results_dir', type=str, help='Folder to save all results into',
                    required=True)
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--checkpoint_path', type=str,
                    help='Name of saved checkpoint to load weights from', required=True)
parser.add_argument('--images_dir', type=str, help='Folder to save all images into',
                    required=True)
parser.add_argument('--pads', nargs='+', type=int, default=[0, 0, 0, 0],
                    help='Padding (top, bottom, left, right)')
parser.add_argument('--face_det_batch_size', type=int,
                    help='Single GPU batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip', default=16)

# parser.add_argument('--resize_factor', default=1, type=int)

args = parser.parse_args()
args.img_size = 96
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
index = [61, 76, 185, 146, 62, 184, 183, 78, 77, 95, 96, 191, 91, 90, 89, 88, 80, 42, 74, 40, 39, 73, 41, 81, 178, 179,
         180, 181, 37, 72, 38, 82, 87, 86, 85, 84, 0, 11, 12, 13, 14, 15, 16, 17, 267, 302, 268, 312, 317, 316, 315,
         314, 405, 404, 403, 402, 311, 271, 303, 269, 270, 304, 272, 310, 318, 319, 320, 321, 409, 408, 407, 415, 324,
         325, 307, 375, 308, 292, 306, 291, 57, 287]

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images):
    batch_size = args.face_det_batch_size

    while 1:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError('Image too big to run face detection on GPU')
            batch_size //= 2
            args.face_det_batch_size = batch_size
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            raise ValueError('Face not detected!')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = get_smoothened_boxes(np.array(results), T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2), True] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    return results

def to_images(dir1, dir2):
    files = list(glob(os.path.join(dir1, "*.mp4")))
    for idx in range(int(len(files)/2)):
        file1 = os.path.join(dir1, "{}.mp4".format(idx))
        file2 = os.path.join(dir1, "{}_origional.mp4".format(idx))

        dir_1 = os.path.join(dir2, str(idx))
        dir_2 = os.path.join(dir2, "{}_origional".format(idx))
        os.makedirs(dir_1)
        os.makedirs(dir_2)
        cmd1 = 'ffmpeg -i {} {}/%04d.jpg'.format(file1, dir_1)
        subprocess.call(cmd1, shell=True)
        cmd2 = 'ffmpeg -i {} {}/%04d.jpg'.format(file2, dir_2)
        subprocess.call(cmd2, shell=True)


def del_ex(dir):
    files = os.listdir(dir)
    for i in range(int(len(files) / 2)):
        path1 = os.path.join(dir, str(i))
        path2 = os.path.join(dir, "{}_origional".format(str(i)))
        images1 = list(glob(os.path.join(path1, "*jpg")))
        images2 = list(glob(os.path.join(path2, "*jpg")))
        length1 = len(images1)
        length2 = len(images2)
        if length1 > length2:
            num = length2
            length2 = length1
            length1 = num
        for k in range(length1+1, length2+1):
            if k <100:
                cmd = "rm {}/00{}.jpg".format(path2, str(k))
            else:
                cmd = "rm {}/0{}.jpg".format(path2, str(k))
            subprocess.call(cmd, shell=True)


index_mouth = [61, 291, 17, 0]
def datagen(frames, landmarks, mel_chunks):
    landmarks_batch, frames_batch, img_batch, coords_batch, true_landmarks_batch, mel_batch = [], [], [], [], [], []
    face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection

    true_landmarks = []
    for img in face_det_results:
        img[0] = cv2.resize(img[0], (96, 96))
        height, width, _ = img[0].shape
        gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
        results = face_mesh.process(gray)
        landmarks_list = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x_vol = []
                y_vol = []
                for i_land in index_mouth:
                    x_vol.append(int((face_landmarks.landmark[i_land].x) * 96))
                    y_vol.append(int((face_landmarks.landmark[i_land].y) * 96))
                x_min, x_max = min(x_vol), max(x_vol)
                y_min, y_max = min(y_vol), max(y_vol)
                landmarks_list.append([x_min, x_max, y_min, y_max])
        true_landmarks.append(landmarks_list)  # seq_length, 4

    for i, mel in enumerate(mel_chunks):
        idx = i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords, valid_frame = face_det_results[idx].copy()
        if not valid_frame:
            continue
        true_landmark = true_landmarks[idx].copy()
        face = cv2.resize(face, (args.img_size, args.img_size))

        landmarks_batch.append(np.asarray(landmarks[idx].detach().numpy()))
        true_landmarks_batch.append(true_landmark)
        frames_batch.append(frame_to_save)
        coords_batch.append(coords)
        img_batch.append(face)
        mel_batch.append(mel.cpu().detach().numpy())

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, landmarks_batch, mel_batch = np.asarray(img_batch), np.asarray(
                landmarks_batch), np.asarray(mel_batch)
            img_masked = img_batch.copy()
            for img_i in range(len(img_batch)):
                if true_landmarks_batch[img_i][0][3] + 6 > 95:
                    true_landmarks_batch[img_i][0][3] = 89
                if true_landmarks_batch[img_i][0][1] + 6 > 95:
                    true_landmarks_batch[img_i][0][1] = 89
                for i_x in range(true_landmarks_batch[img_i][0][2] - 6, true_landmarks_batch[img_i][0][3] + 6):
                    for i_y in range(true_landmarks_batch[img_i][0][0] - 6, true_landmarks_batch[img_i][0][1] + 6):
                        img_batch[img_i][i_x][i_y] = 0
            img_batch = np.concatenate((img_batch, img_masked), axis=3) / 255.
            yield landmarks_batch, img_batch, frames_batch, coords_batch, mel_batch
            img_batch, landmarks_batch, frames_batch, coords_batch, mel_batch = [], [], [], [], []

    if len(img_batch) > 0:
        img_batch, landmarks_batch, mel_batch = np.asarray(img_batch), np.asarray(landmarks_batch), np.asarray(
            mel_batch)
        img_masked = img_batch.copy()
        for img_i in range(len(img_batch)):
            if true_landmarks_batch[img_i][0][3] + 6 > 95:
                true_landmarks_batch[img_i][0][3] = 89
            if true_landmarks_batch[img_i][0][1] + 6 > 95:
                true_landmarks_batch[img_i][0][1] = 89
            for i_x in range(true_landmarks_batch[img_i][0][2] - 6, true_landmarks_batch[img_i][0][3] + 6):
                for i_y in range(true_landmarks_batch[img_i][0][0] - 6, true_landmarks_batch[img_i][0][1] + 6):
                    img_batch[img_i][i_x][i_y] = 0

        img_batch = np.concatenate((img_batch, img_masked), axis=3) / 255.
        yield landmarks_batch, img_batch, frames_batch, coords_batch, mel_batch


fps = 25
mel_step_size = 10
mel_idx_multiplier = 1.9
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                        flip_input=False, device=device)


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Hubert_reconstructor()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


import soundfile as sf
from transformers import AutoProcessor, HubertModel, Wav2Vec2Model
processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
import platform


def main():
    assert args.data_root is not None
    data_root = args.data_root

    if not os.path.isdir(args.results_dir): os.makedirs(args.results_dir)

    with open(args.filelist, 'r') as filelist:
        lines = filelist.readlines()
    model = load_model(args.checkpoint_path)

    for idx, line in enumerate(tqdm(lines)):
        print(line.strip())
        audio_src, video = line.strip().split()

        audio_src = os.path.join(data_root, audio_src) + '.mp4'
        video = os.path.join(data_root, video) + '.mp4'
        out_path = os.path.join(args.results_dir, str(idx)) + '_origional.mp4'
        print(out_path)
        cmd = "cp {} {}".format(video, out_path)
        subprocess.call(cmd, shell=platform.system() != 'Windows')
        print(video)
        print(audio_src)

        command = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'.format(audio_src, '../temp/temp.wav')
        subprocess.call(command, shell=True)
        temp_audio = '../temp/temp.wav'

        wav, sr = sf.read(temp_audio)
        input_values = processor(wav, return_tensors="pt", sampling_rate=16000).input_values
        print(input_values.shape)
        hubert_hidden_state = hubert(input_values).last_hidden_state
        hubert_feature = np.asarray(hubert_hidden_state.squeeze().detach().numpy())
        hubert_feature = hubert_feature.T
        real_len = int((len(wav) / sr) * 25)-1
        print("ok")
        mel_chunks = []
        mel_idx_multiplier = 1.9
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(hubert_feature[0]):
                mel_chunks.append(hubert_feature[:, len(hubert_feature[0]) - mel_step_size:])
                break
            mel_chunks.append(hubert_feature[:, start_idx: start_idx + mel_step_size])
            i += 1
        mel_chunks = mel_chunks[:real_len]
        mel_chunks = torch.FloatTensor(np.array(mel_chunks)).transpose(1, 2).unsqueeze(1)

        video_stream = cv2.VideoCapture(video)

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading or len(full_frames) > len(mel_chunks):
                video_stream.release()
                break
            full_frames.append(frame)
        print(len(full_frames), len(mel_chunks))
        if len(full_frames) < len(mel_chunks):
            print("hehehe")
            continue
        full_frames = full_frames[:len(mel_chunks)]

        try:
            face_det_results = face_detect(full_frames.copy())
        except ValueError as e:
            print(e)
            continue

        landmarks_list = []
        for img in face_det_results:
            img[0] = cv2.resize(img[0], (96, 96))
            gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
            results = face_mesh.process(gray)
            landmarks_origional = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for i_land in index:
                        x_1, y_1 = face_landmarks.landmark[i_land].x, face_landmarks.landmark[i_land].y
                        landmarks_origional.append([x_1, y_1])
            landmarks_list.append(landmarks_origional)
        landmarks_list = torch.FloatTensor(landmarks_list)

        landmarks = torch.FloatTensor(np.asarray(landmarks_list))
        landmarks = landmarks.unsqueeze(1).to(device)
        mel_chunks = mel_chunks.to(device)

        gen = datagen(full_frames.copy(), landmarks.cpu(), mel_chunks)
        print(gen)
        print("abcd")
        for i, (landmarks, img_batch, frames, coords, mel_audio) in enumerate(gen):
            if i == 0:
                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter('../temp/result.avi',
                                      cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            landmarks = torch.FloatTensor(landmarks).to(device)
            print(landmarks.shape)
            mel_batch = torch.FloatTensor(mel_audio).to(device)

            with torch.no_grad():
                pred = model(mel_batch, landmarks, img_batch)
            pred = pred[0].cpu().numpy().transpose(0, 2, 3, 1) * 255.
            for pl, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                pl = cv2.resize(pl.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = pl
                out.write(f)
        out.release()

        vid = os.path.join(args.results_dir, '{}.mp4'.format(idx))

        command = 'ffmpeg -loglevel panic -y -i {} -i {} -strict -2 -q:v 1 {}'.format(temp_audio,
                                                                                      '../temp/result.avi', vid)
        subprocess.call(command, shell=True)
    # to_images(args.results_dir, args.images_dir)
    # del_ex(args.images_dir)

if __name__ == '__main__':
    main()