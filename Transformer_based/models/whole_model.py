from torch import nn
import torch
# from transformers import Wav2Vec2Model


# from conv import Conv1d
from hparams import hparams
from models import lip_former
from models import Reconstructor_lip_audio

class lipformer(nn.Module):
    def __init__(self):
        super(lipformer, self).__init__()

        self.landmarkformer = lip_former()
        self.reconstructor = Reconstructor_lip_audio( )

    def forward(self, audio, landmarks, face):  # landmarks->(B,T,82,2), audio->(B,T,80,16), face->(B, T, 96, 96, 3)
        audio_sequence = audio.squeeze(2)
        output_landmarks = self.landmarkformer(audio_sequence, landmarks)   #B, T, 82, 2
        output = self.reconstructor(output_landmarks, face, audio)
        return output, output_landmarks





