import sounddevice as sd
import scipy.io.wavfile as wav

# Set the recording parameters
duration = 5  # Duration of the recording in seconds
fs = 44100  # Sampling frequency

# Record the audio
print("Recording started. Speak into the microphone...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is complete

# Save the recorded audio as a WAV file
wav.write('recorded_voice.wav', fs, recording)

print("Recording saved as 'recorded_voice.wav'.")
