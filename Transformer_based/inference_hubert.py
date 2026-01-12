from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import hubert_former
import platform
import mediapipe as mp

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str,
                    help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str,
                    help='Filepath of video/image that contains faces to use', required=True)

parser.add_argument('--landmarks', type=str,
                    help='Filepath of video/image that contains faces to use', required=True)

parser.add_argument('--audio', type=str,
                    help='Filepath of video/image that contains faces to use', required=True)

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


def load_model(path):
    model = hubert_former()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


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


from transformers import AutoProcessor, HubertModel, Wav2Vec2Model
processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
import soundfile as sf

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

    model = load_model(args.checkpoint_path)
    face_det_results = face_detect(full_frames)
    origional_landmarks = np.load(args.landmarks)
    landmarks = []
    for img in face_det_results:
        img[0] = cv2.resize(img[0], (96, 96))
        gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
        results = face_mesh.process(gray)
        landmarks_list = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for i_land in index:
                    x_1, y_1 = face_landmarks.landmark[i_land].x, face_landmarks.landmark[i_land].y
                    landmarks_list.append([x_1, y_1])
        landmarks.append(landmarks_list)

    landmarks = torch.FloatTensor(landmarks)

    wav, sr = sf.read(args.audio)
    input_values = processor(wav, return_tensors="pt", sampling_rate=16000).input_values
    print(input_values.shape)
    hubert_hidden_state = hubert(input_values).last_hidden_state
    print(hubert_hidden_state.shape)
    hubert_feature = np.asarray(hubert_hidden_state.squeeze().detach().numpy())
    hubert_feature = hubert_feature.T

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

    mel_chunks= mel_chunks[:75]
    mel_chunks = torch.FloatTensor(np.array(mel_chunks)).transpose(1, 2)
    # mel_chunks = mel_chunks[0:47, :, :]
    mel_chunks = mel_chunks.unsqueeze(1).to(device)
    # mel_chunks = mel_chunks.to(device)
    landmarks = landmarks.unsqueeze(1).to(device)
    print(landmarks.shape)
    landmark_predicted = model(mel_chunks, landmarks).squeeze()
    landmarks = landmarks.squeeze()

    image1 = cv2.imread("image.jpg")
    image1 = cv2.resize(image1, (512, 512))
    for i in range(landmark_predicted.shape[0]):
        image = image1
        for j in range(landmark_predicted.shape[1]):
            x1, y1 = int(landmarks[i][j][0] * 512), int(landmarks[i][j][1] * 512)
            x2, y2 = int(landmark_predicted[i][j][0] * 512), int(landmark_predicted[i][j][1] * 512)
            if x1 == x2 and y1 == y2:
                cv2.circle(image, (x1, y1), 1, (0, 0, 255), -1)
            else:
                cv2.circle(image, (x1, y1), 1, (0, 255, 0), -1)
                cv2.circle(image, (x2, y2), 1, (255, 0, 0), -1)
        cv2.imwrite("result_image/{}.jpg".format(str(i)), image)


if __name__ == '__main__':
    main()
