# import cv2
# import mediapipe as mp
#
# index = [61, 76, 185, 146, 62, 184, 183, 78, 77, 95, 96, 191, 91, 90, 89, 88, 80, 42, 74, 40, 39, 73, 41, 81, 178, 179,
#          180, 181, 37, 72, 38, 82, 87, 86, 85, 84, 0, 11, 12, 13, 14, 15, 16, 17, 267, 302, 268, 312, 317, 316, 315,
#          314, 405, 404, 403, 402, 311, 271, 303, 269, 270, 304, 272, 310, 318, 319, 320, 321, 409, 408, 407, 415, 324,
#          325, 307, 375, 308, 292, 306, 291, 57, 287]
#
# index_mouth = [61, 291, 17, 0]
# img = cv2.imread("img_3.png")
# # img = cv2.resize(img, (96, 96))
#
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh()
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# results = face_mesh.process(gray)
# y_index = []
# x_index = []
# if results.multi_face_landmarks:
#     for landmark in results.multi_face_landmarks:
#         for i in index_mouth:
#             x = int(landmark.landmark[i].x * img.shape[1])
#             y = int(landmark.landmark[i].y * img.shape[0])
#             y_index.append(y)
#             x_index.append(x)
#             cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            # print([x, y])
for i in range(11, 21, 1):
    print(i, i+1)


# for i_x in range(min(y_index) - 6, max(y_index) + 6):
#     for i_y in range(min(x_index) - 6, max(x_index) + 6):
#         img[i_x][i_y] = 0
# #
# cv2.imwrite("image1.jpg", img)

# from glob import glob
# import os
# import numpy as np
# np_list = []
# npy_list = list(glob(os.path.join(r"E:\00001", "*_landmark.npy")))
# for i in npy_list:
#     l = np.load(i)
#     print(l.shape)
#     np_list.append(l)
# npy = np.array(np_list)
# np.save("test.npy", npy)
# import torch
# tens = torch.randn(25, 3, 5, 96, 96)
# tens = tens[:, :, :, :, :tens.shape[4]//2]
# print(tens.shape)
# import cv2
# img = cv2.imread(r"E:\baselines\Wav2Lip\image_2\1.jpg")
# img = cv2.resize(img, (112, 94))
# cv2.imwrite(r"E:\baselines\Wav2Lip\image_2\1.jpg", img)
# import torch
# audio_sequences = torch.randn(1, 1,  80, 16)
# audio = torch.cat([audio_sequences[:, :, i] for i in range(audio_sequences.size(2))], dim=2)
# print(audio.shape)
# audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
# print(audio_sequences.shape)

# import numpy as np
# from glob import glob
# from os.path import join as join
#
# img_names = list(glob(join(r"E:\baselines\Wav2Lip\npy_lrs2\npy_lrs2\5551822044642795487\00002", '*_landmark.npy')))
# imgs_list = []
#
# for img in img_names:
#     landmark = np.load(img)
#     imgs_list.append(landmark)
# imgs_list = np.asarray(imgs_list)
# np.save("00002.npy", imgs_list)
from torch import nn
# import torch
#
#
# tens2 = torch.randn(25, 5, 82, 2)
# landmarks = torch.cat([tens2[i, :, :, :] for i in range(tens2.size(0))], dim=0)
# print(landmarks.shape)

for i in range(11, 21):
    print(i+10, i)

for i in range(11, 21):
    print(i, i-10)