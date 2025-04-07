import whisper
import pyaudio
import wave

# Load the Whisper model
model = whisper.load_model("base")

# Record audio from the microphone
def record_audio(output_file="output.wav", duration=5):
    print("Recording Started.")
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
    initial_prompt = """You are an expert speech therapist. Your goal is to maintain the speech patterns that the user has while you transcribe the audio. In this audio, the user speaks in english. If the user stutters or stammers or takes longs pauses at some words or phrases, add those in the transcript without any corrections. 
    Examples:
    1. Repetitions:
     a. Sound or syllable repetitions: "<<I-I-I>> want to go.", "<<Ba-ba-baby>> is crying."
     b. Whole word repetitions: "<<She she she>> is my friend."
    2. Prolongations:
     a. Stretching out a sound: "<<Ssssssometimes>> I feel nervous.", "<<Mmmmmaybe>> we should leave."
    3. Blocks (Silent or Audible Pauses):
     a. Sudden stops before or during a word: "I want to... (pause)... go.", "He... (tense pause)... left."
    4. Interjections or Fillers (can be normal or excessive): "Um... I was going to the store." "Like... like... I mean..."
    """
    result = model.transcribe(file_path, initial_prompt=initial_prompt)
    print("Result", result)
    print("Transcription: ", result['text'])
    return result

# # Record and transcribe
# record_audio("output.wav", duration=20)
# transcribe_audio("output.wav")
