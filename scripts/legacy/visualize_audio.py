import torch
import torchaudio
import matplotlib.pyplot as plt
import os

def plot_spectrogram(wav_path, out_path):
    import soundfile as sf
    wav, sr = sf.read(wav_path)
    wav = torch.from_numpy(wav).float().unsqueeze(0)
    specgram = torchaudio.transforms.Spectrogram(n_fft=1024)(wav)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(torch.log10(specgram[0] + 1e-9).numpy(), origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram: {os.path.basename(wav_path)}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    wav_file = "outputs/vocoder_samples/epoch_6.wav"
    out_img = "outputs/vocoder_samples/epoch_6_spec.png"
    if os.path.exists(wav_file):
        plot_spectrogram(wav_file, out_img)
        print(f"Spectrogram saved to {out_img}")
    else:
        print(f"Error: {wav_file} not found")
