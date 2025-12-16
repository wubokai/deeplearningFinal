from torch import nn
import torch


from hparams import hparams


class landmark_former(nn.Module):
    def __init__(self):
        super(landmark_former, self).__init__()

        self.mfcc_dim = nn.Linear(1280, 512)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4,batch_first=True)
        self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.landmarks_encoder = nn.Linear(hparams.landmarks_num, 512)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hparams.decoder_feature_dim, nhead=4, batch_first=True)
        self.landmarks_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.output_layer = nn.Linear(512, hparams.landmarks_num)
        self.act = nn.ReLU()
        self.layer_norm = nn.LayerNorm(164)

    def mfcc_up(self, audio):
        mfcc = self.mfcc_dim(audio)
        return mfcc

    def forward(self, audio, landmarks):  # landmarks->(B,5,856), audio->(B, 5, 80, 16)
        audio = torch.cat([audio[:, :, i] for i in range(audio.size(2))], dim=2)
        audio_feature = self.mfcc_up(audio)
        audio_embedding = self.audio_encoder(audio_feature)
        landmarks = torch.cat([landmarks[:, :, i] for i in range(landmarks.size(2))], dim=2)
        landmarks_embedding = self.landmarks_encoder(landmarks)
        decoder_output = self.landmarks_decoder(landmarks_embedding, audio_embedding)
        output = self.output_layer(decoder_output)
        output = self.layer_norm(output)
        output = self.act(output)
        output = torch.split(output, split_size_or_sections=82, dim=2)
        output = torch.cat((output[0].unsqueeze(3), output[1].unsqueeze(3)), dim=3)

        return output




