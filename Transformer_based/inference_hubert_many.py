from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Reconstructor_hubert_wav2lip
import platform
import mediapipe as mp

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str,
                    help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--landmarks', type=str,
                    help='Filepath of video/audio file to use as raw audio source', required=True)


parser.add_argument('--face', type=str,
                    help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--audio', type=str,
                    help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                    default='results/2.mp4')

parser.add_argument('--static', type=bool,
                    help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int,
                    help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int,
                    help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                         'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                         'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                         'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
index = [61, 76, 185, 146, 62, 184, 183, 78, 77, 95, 96, 191, 91, 90, 89, 88, 80, 42, 74, 40, 39, 73, 41, 81, 178, 179,
         180, 181, 37, 72, 38, 82, 87, 86, 85, 84, 0, 11, 12, 13, 14, 15, 16, 17, 267, 302, 268, 312, 317, 316, 315,
         314, 405, 404, 403, 402, 311, 271, 303, 269, 270, 304, 272, 310, 318, 319, 320, 321, 409, 408, 407, 415, 324,
         325, 307, 375, 308, 292, 306, 291, 57, 287]

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    args.static = True


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


mel_step_size = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path1):
    model1 = Reconstructor_hubert_wav2lip()
    print("Load checkpoint from: {}".format(path1))
    checkpoint1 = _load(path1)
    s1 = checkpoint1["state_dict"]
    new_s1 = {}
    for k1, v1 in s1.items():
        new_s1[k1.replace('module.', '')] = v1
    model1.load_state_dict(new_s1)

    model1 = model1.to(device)
    return model1.eval()


def face_detect(images):  # 截取人脸
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    batch_size = args.face_det_batch_size

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results


index_mouth = [61, 291, 17, 0]
from transformers import AutoProcessor, HubertModel, Wav2Vec2Model
processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
import soundfile as sf


def datagen(frames, landmarks, mel_chunks):
    landmarks_batch, frames_batch, img_batch, coords_batch, true_landmarks_batch, mel_batch = [], [], [], [], [], []
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

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
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()
        true_landmark = true_landmarks[idx].copy()
        face = cv2.resize(face, (args.img_size, args.img_size))

        landmarks_batch.append(np.asarray(landmarks[idx]))
        true_landmarks_batch.append(true_landmark)
        frames_batch.append(frame_to_save)
        coords_batch.append(coords)
        img_batch.append(face)
        mel_batch.append(mel)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, landmarks_batch, mel_batch = np.asarray(img_batch), np.asarray(
                landmarks_batch)/96., np.asarray(mel_batch)

            img_masked = img_batch.copy()
            for img_i in range(len(img_batch)):
                for i_x in range(true_landmarks_batch[img_i][0][2] - 6, true_landmarks_batch[img_i][0][3] + 6):
                    for i_y in range(true_landmarks_batch[img_i][0][0] - 6, true_landmarks_batch[img_i][0][1] + 6):
                        img_batch[img_i][i_x][i_y] = 0
            print(mel_batch.shape)
            img_batch = np.concatenate((img_batch, img_masked), axis=3) / 255.
            yield img_batch, landmarks_batch, frames_batch, coords_batch, mel_batch
            img_batch, landmarks_batch, frames_batch, coords_batch, mel_batch = [], [], [], [], []

    if len(img_batch) > 0:
        img_batch, landmarks_batch, mel_batch = np.asarray(img_batch), np.asarray(landmarks_batch)/96., np.asarray(
            mel_batch)
        img_masked = img_batch.copy()

        for img_i in range(len(img_batch)):
            for i_x in range(true_landmarks_batch[img_i][0][2] - 6, true_landmarks_batch[img_i][0][3] + 6):
                for i_y in range(true_landmarks_batch[img_i][0][0] - 6, true_landmarks_batch[img_i][0][1] + 6):
                    img_batch[img_i][i_x][i_y] = 0
        print(mel_batch.shape)
        img_batch = np.concatenate((img_batch, img_masked), axis=3) / 255.
        yield landmarks_batch, img_batch, frames_batch, coords_batch, mel_batch


def main():
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    print("Number of frames available for inference: " + str(len(full_frames)))

    wav1, sr = sf.read(args.audio)
    input_values = processor(wav1, return_tensors="pt", sampling_rate=16000).input_values
    hubert_hidden_state = hubert(input_values).last_hidden_state

    hubert_feature = np.asarray(hubert_hidden_state.squeeze().detach().numpy())
    hubert_feature = hubert_feature.T
    landmark = np.load(args.landmarks)
    len_landmark = len(landmark)
    print(len_landmark)
    print(hubert_feature.shape)
    mel_chunks = []
    mel_idx_multiplier = 1.8
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(hubert_feature[0]):
            mel_chunks.append(hubert_feature[:, len(hubert_feature[0]) - mel_step_size:])
            break
        mel_chunks.append(hubert_feature[:, start_idx: start_idx + mel_step_size])
        i += 1

    batch_size = args.wav2lip_batch_size
    mel_chunks = mel_chunks[0:len_landmark]
    gen = datagen(full_frames.copy(), landmark, mel_chunks)
    mel_length = len(mel_chunks)
    for i, (landmarks, img_batch, frames, coords, mel_audio) in enumerate(tqdm(gen,
                                                                               total=int(
                                                                                   np.ceil(float(
                                                                                       mel_length) / batch_size)))):
        if i == 0:
            model1 = load_model(args.checkpoint_path)
            print("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        landmarks = torch.FloatTensor(np.transpose(landmarks, (0, 1, 2))).unsqueeze(0).to(device)
        mel_audio = torch.FloatTensor(np.transpose(mel_audio, (0, 2, 1))).unsqueeze(1).to(device)
        print(mel_audio.shape)
        if len(img_batch.size()) > 4:
            mel_audio = mel_audio.unsqueeze(1)

        with torch.no_grad():
            pred = model1(landmarks, img_batch, mel_audio)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')


if __name__ == '__main__':
    main()
