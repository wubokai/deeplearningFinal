import librosa
import librosa.filters
# `librosa` 是一个用于音频和音乐分析的 Python 库，支持加载音频文件、特征提取（如 Mel 频谱）、音调和节奏分析等功能。
# `librosa.filters` 提供了音频处理中的各种滤波器函数，例如 Mel 滤波器等。
import numpy as np
# `numpy` 是一个用于科学计算的库，提供了多维数组支持以及大量数学运算函数，广泛应用于数据处理和数值计算。
# import tensorflow as tf
# `tensorflow` 是一个深度学习框架，支持大规模的机器学习和神经网络建模，虽然此行代码被注释掉了，但其主要功能是用于创建和训练神经网络。
from scipy import signal
# `scipy.signal` 是 SciPy 库中的信号处理模块，包含了各种滤波器设计、信号变换、卷积、去噪等功能，用于处理和分析时域和频域的信号。
from scipy.io import wavfile
# `scipy.io.wavfile` 提供了读取和写入 WAV 音频文件的功能，可以加载和保存标准的 PCM 编码音频数据。
from hparams import hparams as hp
# `hparams` 模块通常用于存储参数（hyperparameters），如采样率、滤波器尺寸、模型架构等。在此代码中，`hparams` 被导入并重命名为 `hp`，表示它存储了程序的关键配置参数。


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size


def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db

    if hp.signal_normalization:
        return _normalize(S)
    return S


def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db

    if hp.signal_normalization:
        return _normalize(S)
    return S


def _lws_processor():
    import lws
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")


def _stft(y):
    if hp.use_lws:
        return _lws_processor(hp).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_size)


##########################################################
# Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################
# Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


# Conversions
_mel_basis = None


def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,
                               fmin=hp.fmin, fmax=hp.fmax)


def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)

    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))


def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value,
                              hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value))
                    + hp.min_level_db)
        else:
            return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)

    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
