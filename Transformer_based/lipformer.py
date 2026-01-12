from torch import nn
import torch
from transformers import Wav2Vec2Model


from models.conv import Conv1d, Conv1dTranspose
from hparams import hparams


class lip_former(nn.Module):
    def __init__(self):
        super(lip_former, self).__init__()

        self.mfcc_dim = nn.Linear(hparams.num_mels, hparams.encoder_feature_dim)
        self.audio_encoder = Wav2Vec2Model.from_pretrained("./wav2vec2.0")
        self.audio_encoder.feature_extractor._freeze_parameters()

        self.landmarks_encoder = nn.Sequential(
            Conv1dTranspose(hparams.landmarks_num, hparams.decoder_feature_dim, kernel_size=1),
            Conv1d(hparams.decoder_feature_dim, hparams.decoder_feature_dim, residual=True)
        )

        decoder_layer = nn.TransformerDecoderLayer(d_model=hparams.decoder_feature_dim, nhead=4, batch_first=True)
        self.landmarks_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.output_layer = Conv1d(hparams.decoder_feature_dim, hparams.landmarks_num)

    def mfcc_up(self, audio):
        mfcc = self.mfcc_dim(audio)
        return mfcc

    def forward(self, audio, landmarks):  # landmarks->(B,5,82,2), audio->(B,80,16)
        audio_feature = self.mfcc_up(audio)
        s1, s2, s3 = audio_feature.shape
        audio_feature = torch.reshape(audio_feature, (s1*s2, s3))

        audio_hidden_states = self.audio_encoder(audio_feature).last_hidden_state   #->800, 1, 768
        print(audio_hidden_states.shape, audio_hidden_states.squeeze(1).shape)
        audio_hidden_states = torch.reshape(audio_hidden_states.squeeze(1), (s1, s2, 768))
        print(audio_hidden_states.shape)
        landmarks_features = self.landmarks_encoder(landmarks)  #->32, 25, 768
        landmarks_features = torch.transpose(landmarks_features, 1, 2)
        print(landmarks_features.shape)

        landmarks_outputs_hidden_states = self.landmarks_decoder(audio_hidden_states, landmarks_features)

        landmarks_outputs = self.output_layer(torch.transpose(landmarks_outputs_hidden_states, 1, 2))

        return landmarks_outputs_hidden_states, landmarks_outputs







