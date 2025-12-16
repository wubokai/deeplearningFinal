import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .conv import Conv2dTranspose, Conv1d, Conv2d


class Reconstructor_hubert_wav2lip_sum(nn.Module):
    def __init__(self):
        super(Reconstructor_hubert_wav2lip_sum, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),  # 96,96

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 48,48
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24,24
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 12,12
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 6,6
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                          Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        self.lip_encoder = nn.Sequential(
            Conv1d(2, 4, 3, 1, 1),  # 37
            Conv1d(4, 4, 3, 1, 1, residual=True),
            Conv1d(4, 4, 3, 1, 1, residual=True),

            Conv1d(4, 8, 3, 1, 1),  # 37
            Conv1d(8, 8, 3, 1, 1, residual=True),
            Conv1d(8, 8, 3, 1, 1, residual=True),

            Conv1d(8, 16, 3, 2, 1),  # 37
            Conv1d(16, 16, 3, 1, 1, residual=True),
            Conv1d(16, 16, 3, 1, 1, residual=True),

            Conv1d(16, 32, 3, 2, 1),  # 19
            Conv1d(32, 32, 3, 1, 1, residual=True),
            Conv1d(32, 32, 3, 1, 1, residual=True),

            Conv1d(32, 64, 3, 2, 1),  # 10
            Conv1d(64, 64, 3, 1, 1, residual=True),
            Conv1d(64, 64, 3, 1, 1, residual=True),

            Conv1d(64, 128, 3, 2, 1),  # 5
            Conv1d(128, 128, 3, 1, 1, residual=True),
            Conv1d(128, 128, 3, 1, 1, residual=True),

            Conv1d(128, 256, 3, 2, 1),  # 3
            Conv1d(256, 256, 3, 1, 1, residual=True),
            Conv1d(256, 256, 3, 1, 1, residual=True),

            Conv1d(256, 512, 3, 1, 0),  # 1
            Conv1d(512, 512, 1, 1, 0, residual=True),
        )
        self.audio_encoder = nn.Sequential(
            Conv1d(1024, 512, kernel_size=2, stride=1, padding=1),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv1d(512, 512, kernel_size=1, stride=1, padding=0),
             )

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0), ),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),  # 3,3
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),  # 6, 6

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True), ),  # 12, 12

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), ),  # 24, 24

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), ),  # 48, 48

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), ), ])  # 96,96

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())

    def forward(self, lip_sequences, face_sequences, audio_sequences): # B, T, 1024
        B = lip_sequences.size(0)

        lip_sequences = torch.cat([lip_sequences[i, :, :, :] for i in range(lip_sequences.size(0))], dim=0)
        lip_sequences = lip_sequences.transpose(1, 2)


        audio_sequences = audio_sequences.transpose(1, 2)
        length = audio_sequences.shape[2]
        audio_chunks = []
        for i in range(0, length, 2):
            if i + 2 > length:
                break
            audio_chunks.append(audio_sequences[:, :, i:2 + i])
        audio_chunks = torch.stack(audio_chunks)
        audio_sequences = torch.cat([audio_chunks[:, i, :, :] for i in range(audio_chunks.size(1))], dim=0)
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        lip_embedding = self.lip_encoder(lip_sequences)  # B, 512, 1, 1

        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.unsqueeze(3)
        lip_embedding = lip_embedding.unsqueeze(3)
        joint_embedding = audio_embedding + lip_embedding

        feats = []
        x = face_sequences

        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = joint_embedding

        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                raise e

            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)  # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)

        else:
            outputs = x

        return outputs
