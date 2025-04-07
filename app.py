import streamlit as st
import sys
import os
from datetime import datetime

# Directory to save recordings and transcriptions
os.makedirs("recordings", exist_ok=True)
os.makedirs("transcripts", exist_ok=True)

# Add src/scripts to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "scripts"))

from llm_wrapper import LLMClient
from speech_to_text import record_audio, transcribe_audio

st.title("🗣️ SpeechFlowAI - Fluency Analysis")

# Step 1: Get a paragraph from LLM
if st.button("📄 Generate Paragraph to Read"):
    client = LLMClient(model="llama2")
    prompt = "You are a speech therapist agent that can analyses users speech and suggest exercises. First give the user a complicated paragraph to read, it should have 'h' 'm' words. Keep it around 60 words."
    paragraph = client.generate(prompt)
    st.session_state.paragraph = paragraph

if "paragraph" in st.session_state:
    st.markdown("### 📝 Please read this aloud:")
    st.info(st.session_state.paragraph)
    
    # Step 2: Record user speech
    if st.button("🎤 Record Speech (20 sec)"):
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        audio_path = f"recordings/recording_{timestamp}.wav"
        transcript_path = f"transcripts/transcript_{timestamp}.txt"

        st.info(f"Recording started")
        record_audio(output_file=audio_path, duration=20)
        # st.success("Recording completed!")
        st.success(f"Recording saved: {audio_path}")

        # Step 3: Transcribe
        with st.spinner("🧠 Transcribing..."):
            result = transcribe_audio(audio_path)
            transcript = result['text']
            st.session_state.transcript = transcript

            # Save transcript
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)

            st.success("Transcription complete!")

if "transcript" in st.session_state:
    st.markdown("### 🗒️ Transcription:")
    st.code(st.session_state.transcript)

    # Step 4: Analyze
    if st.button("🧠 Analyze for Fluency Issues"):
        analysis_prompt = f"""
        You are a speech therapist agent that can analyses users speech and suggest exercises.
        Analyze this speech for fluency issues (e.g., stuttering, cluttering, prolongations, etc.).
        Provide detailed explanation on which words or patterns users has a problem and suggest exercises to improve those.
       
        Transcript:
        {st.session_state.transcript}
        """
        client = LLMClient(model="llama2")
        analysis = client.generate(analysis_prompt)
        st.markdown("### 🩺 Analysis:")
        st.write(analysis)