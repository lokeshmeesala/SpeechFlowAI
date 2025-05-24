import whisper
import pyaudio
import wave
import streamlit as st
import time

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
    st.info(f"🎙️ Recording for {duration} seconds... Speak clearly into the mic.")
    progress_bar = st.progress(0)
    
    frames = []
    
    total_chunks = int(rate / chunk * duration)
    for i in range(0, total_chunks):
        data = stream.read(chunk)
        frames.append(data)
        progress = int((i + 1) / total_chunks * 100)
        progress_bar.progress(progress)
        time.sleep(chunk / rate)

    st.success("✅ Recording finished.")
    
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

    # initial_prompt = """The user is speaking English with stuttering or disfluent patterns. Transcribe exactly as spoken."""
    result = model.transcribe(file_path, initial_prompt=initial_prompt, language="en")
    print("Result", result)
    print("Transcription: ", result['text'])
    return result

# # Record and transcribe
# record_audio("output.wav", duration=20)
# transcribe_audio("output.wav")
