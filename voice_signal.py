import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Load the recorded voice from the WAV file
fs, voice_data = wav.read('recorded_voice.wav')

# Normalize the signal
voice_signal = voice_data / np.max(np.abs(voice_data))

# Create the time axis
duration = len(voice_signal) / fs
t = np.linspace(0, duration, len(voice_signal))

def record_voice():
    # Plot the voice signal
    plt.figure()
    plt.plot(t, voice_signal)
    plt.title('Recorded Voice Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

# Generate noise signal
noise_signal = np.random.normal(0, 0.1, len(voice_signal))

# Add noise to the voice signal
noisy_signal = voice_signal + noise_signal

# Apply a highpass filter to remove low-frequency noise
cutoff_freq = 500  # Cutoff frequency in Hz
b, a = signal.butter(4, cutoff_freq / (fs / 2), 'highpass')
filtered_signal = signal.lfilter(b, a, noisy_signal)

def filter_voice():
    # Plot the signals
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, noisy_signal)
    plt.title('Noisy Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(t, filtered_signal)
    plt.title('Filtered Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

def signal_init():
    while True:
        print("1. Record voice")
        print("2. Filter voice")
        print("3. Exit")

        choice = int(input("Enter your choice: "))

        match choice:
            case 1:
                record_voice()
            case 2:
                filter_voice()
            case 3:
                return
            case _:
                print("Invalid choice. Try again.")
        
if __name__ == "__main__":
    signal_init()