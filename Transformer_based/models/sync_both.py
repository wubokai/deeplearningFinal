import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2d, Conv1d


class SyncNet_both(nn.Module):
    def __init__(self):
        super(SyncNet_both, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

        self.lip_encoder = nn.Sequential(
            Conv1d(10, 16, 3, 2, 1),  # 37
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
            Conv1d(512, 512, 1, 1, 0),
        )


    def forward(self, audio_sequences, face_sequences, lip_sequences):  # audio_sequences := (B, dim, T) , # lip_sequences :=(B, 5, 428, 2)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)
        lip_sequences = lip_sequences.permute(0, 1, 3, 2)
        lip_sequences = torch.cat([lip_sequences[:, i, :, :] for i in range(lip_sequences.size(1))], dim=1)
        # lip_sequences = torch.cat([lip_sequences[:, :, i, :] for i in range(lip_sequences.size(2))], dim=2)
        lip_embedding = self.lip_encoder(lip_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)
        lip_embedding = lip_embedding.view(lip_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        lip_embedding = F.normalize(lip_embedding, p=2, dim=1)



        return audio_embedding, face_embedding, lip_embedding


