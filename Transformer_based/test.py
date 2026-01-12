import torch.nn as nn
import torch


# tens1 = torch.randn(32, 5, 82, 2)
# tens1 = tens1.reshape(32, 5*82*2)
#
# tens2 = torch.randn(32, 5, 82, 2)
# tens2 = tens2.reshape(32, 5*82*2)
#
# x = nn.functional.cosine_similarity(tens1, tens2)
# print(x)
# mse = nn.MSELoss()
# loss = mse(tens1, tens2)
# print(loss, loss.shape)

# import numpy as np
#
# landmark = np.load(r"E:\baselines\Wav2Lip\npy_lrs2\npy_lrs2\5551822044642795487\00007\21_landmark.npy")
# print(np.max(landmark))
# from models import Conv2d
#
# face_encoder_blocks = nn.ModuleList([
#             nn.Sequential(Conv2d(3, 16, kernel_size=7, stride=1, padding=3)),  # 96,96
#
#             nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 48,48
#                           Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
#                           Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),
#
#             nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24,24
#                           Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
#                           Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
#                           Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),
#
#             nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 12,12
#                           Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
#                           Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),
#
#             nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 6,6
#                           Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
#                           Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),
#
#             nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 3,3
#                           Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),
#
#             nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
#                           Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])
#
# x = torch.randn(160, 3, 96, 96)
# for f in face_encoder_blocks:
#     x = f(x)
#     print(x.shape)
# import torch
#
# tens1 = torch.randn(512, 1,1)
# tens2 = torch.randn(512, 1,1)
# tens3 = torch.cat((tens1, tens2), dim=0)
# print(tens3.shape)

# from transformers import AutoProcessor, HubertModel, Wav2Vec2Processor
# import soundfile as sf
# #
# # processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
# processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
# # audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
# model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
# #
# # import torch
# wav, sr = sf.read("audio.wav")
# # wav = torch.randn(25, 25600)
# #
# input_values = processor(wav, return_tensors="pt", sampling_rate=16000).input_values  # Batch size 1
# # print(input_values)
# # # input_values = torch.randn(25, 25600)
# hidden_states = model(input_values).last_hidden_state
# print(hidden_states)
# # h = audio_encoder(input_values).last_hidden_state
# print(hidden_states.shape)
import numpy as np
#
#
# def crop_audio_window(spec, start_frame):
#     # num_frames = (T x hop_size * fps) / sample_rate
#     start_frame_num = start_frame
#     start_idx = int(80. * (start_frame_num / float(25)))
#
#     end_idx = start_idx + 16
#
#     return spec[start_idx: end_idx, :]
#
# def get_segmented_mels(self, spec, start_frame):
#     mels = []
#     start_frame_num = self.get_frame_id(start_frame) + 1  # 0-indexing ---> 1-indexing
#     if start_frame_num - 2 < 0: return None
#     for i in range(start_frame_num, start_frame_num + 5):
#         m = self.crop_audio_window(spec, i - 2)
#         if m.shape[0] != 16:
#             return None
#         mels.append(m.T)
#
#     mels = np.asarray(mels)
# mel = np.load("audio.npy")
# a = crop_audio_window(mel, 3)
# a= np.array(a)
# print(a.shape)
# print(mel.shape)
import torch
class Conv1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, act='ReLU', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm1d(cout)
        )
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'Tanh':
            self.act = nn.Tanh()
        self.residual = residual


    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        out = self.act(out)
        return out

# con = nn.Sequential(
#             Conv1d(1024, 128, kernel_size=3, stride=1, padding=1),
#
#             Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
#             Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
#
#             Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
#             Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
#
#             Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
#             Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
#
#             Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
#             Conv1d(512, 512, kernel_size=3, stride=1, padding=0),
#             Conv1d(512, 512, kernel_size=1, stride=1, padding=0),
#              )
#
tens1 = torch.randn(25, 1024, 6)
#
# ouptut = con(tens1)
# print(ouptut.shape)

length = tens1.shape[2]
tens = []
for i in range(0, length, 2):
    if i+2 > length:
        break
    tens.append(tens1[:, :, i:2+i])
tens = torch.stack(tens)
print(tens.shape)
tens = torch.cat([tens[:, i, :, :] for i in range(tens.size(1))], dim=0)
print(tens.shape)

