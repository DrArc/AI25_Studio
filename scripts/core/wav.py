# scripts/wav.py
import numpy as np
from scipy.io import wavfile

def wav_to_laeq_dba(wav_path):
    samplerate, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data / (np.max(np.abs(data)) + 1e-9)
    rms = np.sqrt(np.mean(data**2))
    ref = 1.0
    spl = 20 * np.log10(rms / ref + 1e-9)
    return round(spl, 2)
