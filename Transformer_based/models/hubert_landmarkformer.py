from torch import nn
import torch


from hparams import hparams
from .Positional_encoding import PositionalEncoder
from .conv import Conv1d


class hubert_former(nn.Module):
    def __init__(self):
        super(hubert_former, self).__init__()
        self.Positional_Encoding1 = PositionalEncoder(d_model=1024)
        self.Positional_Encoding2 = PositionalEncoder(d_model=512)

        self.audio_encoder = nn.Sequential(
            Conv1d(1024, 128, kernel_size=3, stride=1, padding=1),

            Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv1d(512, 512, kernel_size=1, stride=1, padding=0), )

        self.landmarks_encoder = nn.Sequential(
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
        decoder_layer = nn.TransformerDecoderLayer(d_model=hparams.decoder_feature_dim, nhead=4, batch_first=True)
        self.landmarks_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.output_layer = nn.Linear(512, hparams.landmarks_num)
        self.act = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(512)


    def forward(self, audio, landmarks):  # landmarks->(B,5,82,2), audio->(B, 5, 10, 1024)
        B, T = landmarks.shape[0], landmarks.shape[1]
        audio = torch.cat([audio[:, i, :, :] for i in range(audio.size(1))], dim=0)
        audio_feature = self.Positional_Encoding1(audio)
        audio_feature = audio_feature.transpose(1, 2)

        audio_embedding = self.audio_encoder(audio_feature)
        audio_embedding = audio_embedding.squeeze()
        audio_embedding = self.layer_norm1(audio_embedding)
        print(audio_embedding.shape)
        audio_embedding = torch.reshape(audio_embedding, (B, T, 512))

        landmarks = torch.cat([landmarks[i, :, :, :] for i in range(landmarks.size(0))], dim=0)
        landmarks = landmarks.transpose(1, 2)
        landmarks_embedding = self.landmarks_encoder(landmarks)
        landmarks_embedding = landmarks_embedding.squeeze()
        landmarks_embedding = torch.reshape(landmarks_embedding, (B, T, 512))
        landmarks_embedding = self.layer_norm2(landmarks_embedding)
        landmarks_embedding = self.Positional_Encoding2(landmarks_embedding)

        decoder_output = self.landmarks_decoder(landmarks_embedding, audio_embedding)
        output = self.output_layer(decoder_output)
        output = self.act(output)
        output = torch.split(output, split_size_or_sections=82, dim=2)
        output = torch.cat((output[0].unsqueeze(3), output[1].unsqueeze(3)), dim=3)

        return output

