import whisper
import pyaudio
import wave

# Load the Whisper model
model = whisper.load_model("base")

# Record audio from the microphone
def record_audio(output_file="output.wav", duration=5):
    rate = 16000
    chunk = 1024
    channels = 1
    format = pyaudio.paInt16
    p = pyaudio.PyAudio()

    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    
    print(f"Recording for {duration} seconds...")
    
    frames = []
    
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    print("Recording finished.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

# Transcribe recorded audio
def transcribe_audio(file_path="output.wav"):
    result = model.transcribe(file_path)
    print("Transcription: ", result['text'])

# Record and transcribe
record_audio("output.wav", duration=20)  # Record for 5 seconds
transcribe_audio("output.wav")
