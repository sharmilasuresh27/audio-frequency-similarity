import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# Load audio files
ref_audio, sr1 = librosa.load(r"D:\audio_similarity_project\audio1.wav", sr=None)
pat_audio, sr2 = librosa.load(r"D:\audio_similarity_project\audio2.wav", sr=None)

# Make both signals same length
min_len = min(len(ref_audio), len(pat_audio))
ref_audio = ref_audio[:min_len]
pat_audio = pat_audio[:min_len]

# Time axis
time = np.linspace(0, min_len / min(sr1, sr2), min_len)


# Plot time-domain comparison (LIKE YOUR IMAGE)
plt.figure(figsize=(10, 5))
plt.plot(time, ref_audio, label="Reference (ref)", color="blue")
plt.plot(time, -pat_audio, label="Pattern (pat)", color="orange")  # inverted for clarity

plt.title("Audio Signal Comparison")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- Similarity Score (Frequency Domain) --------
fft_ref = np.abs(np.fft.fft(ref_audio))
fft_pat = np.abs(np.fft.fft(pat_audio))

fft_ref = fft_ref[:min_len // 2]
fft_pat = fft_pat[:min_len // 2]
# Frequency-domain plot (FINAL REQUIRED GRAPH)
plt.figure(figsize=(10, 5))
plt.plot(fft_ref, label="Reference Audio (FFT)", alpha=0.7)
plt.plot(fft_pat, label="Pattern Audio (FFT)", alpha=0.7)

plt.title("Frequency Domain Comparison")
plt.xlabel("Frequency Bins")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


similarity_score = 1 - cosine(fft_ref, fft_pat)
print(f"Similarity Score: {similarity_score:.4f}")
