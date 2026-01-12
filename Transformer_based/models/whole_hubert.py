from torch import nn
import torch

from .hubert_landmarkformer import hubert_former
from .reconstruct_hubert_wav2lip import Reconstructor_hubert_wav2lip


class Hubert_reconstructor(nn.Module):
    def __init__(self):
        super(Hubert_reconstructor, self).__init__()

        self.landmark_former = hubert_former()
        self.reconstructor = Reconstructor_hubert_wav2lip()

    def forward(self, audio, landmarks, face):  # landmarks->(B,T,82,2), audio->(B,T,80,16), face->(B, T, 96, 96, 3)
        output_landmarks = self.landmark_former(audio, landmarks)   #B, T, 82, 2
        output = self.reconstructor(output_landmarks, face, audio)
        return output, output_landmarks





