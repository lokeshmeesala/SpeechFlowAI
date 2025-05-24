import streamlit as st
from audiorecorder import audiorecorder
import sys
import os
from datetime import datetime

# Directory to save recordings and transcriptions
os.makedirs("recordings", exist_ok=True)
os.makedirs("transcripts", exist_ok=True)

# Add src/scripts to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "scripts"))

from llm_wrapper import LLMClient
from speech_to_text import transcribe_audio

st.title("🗣️ SpeechFlowAI - Fluency Analysis")
# Initialize session state variables
if "generate_paragraph" not in st.session_state:
    st.session_state.generate_paragraph = False
if "read_past_history" not in st.session_state:
    st.session_state.read_past_history = False
if "paragraph" not in st.session_state:
    st.session_state.paragraph = ""

col1, col2 = st.columns(2)

# Button 1: Generate paragraph
with col1:
    if st.button("📄 Generate Paragraph to Read"):
        st.session_state.generate_paragraph = True
        client = LLMClient(model="llama2")
        prompt = """You are a speech therapist agent that helps assess speech fluency. Your task is to generate a paragraph that challenges the user’s fluency, articulation, and rhythm. The paragraph must meet the following conditions:

- Length: Between 15 and 20 words (strictly no more than 25 words)
- Content: Include multisyllabic words, natural pausing points, and flowing connected speech
- Tone: Use realistic and natural language, as seen in clinical assessments like the Rainbow or Grandfather Passage
- Output: Your output should contain only the paragraph. Do NOT include any introductions, explanations, or concluding statements like 'Sure! Here is a paragraph that challenges the user's fluency, articulation, and rhythm'; 'Okay! Here is a paragraph'; 'Please read the below' etc.
- Clarity: No lists or markdown formatting — return plain text only

Few shot Examples:

Assistant: The highway was hidden behind the hills, humming with heavy morning traffic. Michael hesitated before merging, holding the wheel with both hands as his heart hammered in his chest.

Assistant: She silently slid the silver spoon into the steaming stew, savoring the scent of simmering spices while listening to soft static from the speakers in the corner.

Assistant: Olivia organized an orchestra of octopuses for an outrageous ocean opera. The melody murmured through the massive marina, mesmerizing the murmuring masses on the mossy pier.
"""
        st.session_state.paragraph = client.generate(prompt)

# Show paragraph if triggered
if st.session_state.generate_paragraph:
    st.markdown("### 📝 Please read this aloud:")
    st.info(st.session_state.paragraph)
    
    # Step 2: Record user speech
    if st.button("🎤 Start Recording Speech"):
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        audio_path = f"recordings/recording_{timestamp}.wav"
        transcript_path = f"transcripts/transcript_{timestamp}.txt"

        st.info(f"Recording started")
        average_words_per_sec = 2.5
        paragraph_len = len(st.session_state.paragraph.split())
        tentative_duration = paragraph_len/average_words_per_sec
        # record_audio(output_file=audio_path, duration=tentative_duration)
        audio = audiorecorder("Click to record", "Recording...")
        if len(audio) > 0:
            st.audio(audio.tobytes(), format="audio/wav")

            # Optionally save the audio to file
            with open(audio_path, "wb") as f:
                f.write(audio.tobytes())

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
            You are a speech therapist agent that analyzes users' speech for fluency issues and provides detailed, actionable exercises to improve their speech.

Your task:
1. Analyze the transcript carefully for fluency issues such as:
   - Repetitions (e.g., "I-I-I want to go")
   - Prolongations (e.g., "Ssssometimes")
   - Cluttering (words run together or speech is too fast)
   - Blocks or pauses (e.g., "I want to... (pause)... go")
   - Fillers (e.g., "um", "uh", "like")

2. Clearly identify where the user struggles (specific words/phrases or patterns).
3. Suggest **targeted exercises** that a speech therapist would actually assign, such as:
   - Pacing exercises (e.g., using a metronome)
   - Easy onset voice drills
   - Mirror reading
   - Word chunking
   - Pausing and breathing techniques
   - Practicing specific syllables or consonant transitions

4. Format your response like this:
---
**Fluency Issues Detected**:
- [Issue 1] → Explanation
- [Issue 2] → Explanation

**Suggested Exercises**:
- [Exercise 1 Title]: Instructions
- [Exercise 2 Title]: Instructions

Example: **Fluency Issues Detected**:
- Prolongation on the phrase “s-s-streaming sunshine”: The user extends the “s” sound excessively.
- Repetition in “the-the-the hills”: A classic example of sound repetition.

**Suggested Exercises**:
- Easy Onset for S Sounds: Practice starting ‘s’ words with a soft airflow. For example, take a small breath and gently glide into the word “sunshine.” Repeat with a mirror 10 times daily.
- Word Chunking: Break sentences into small chunks. Practice “The hills | are streaming | with sunshine.” Pause at each bar. This improves pacing and reduces repetition.

Use clinical tone. Do NOT provide general advice like "practice regularly". Focus on **what to practice, how, and why**.
        
            Original Text that user read:
            {st.session_state.paragraph}
            
            Transcript From User's Recording:
            {st.session_state.transcript}
            """
            client = LLMClient(model="llama2")
            analysis = client.generate(analysis_prompt)
            st.markdown("### 🩺 Analysis:")
            st.write(analysis)

with col2:
    if st.button("📚 Show My Progress + Suggested Exercises"):
        st.session_state.paragraph = "To Be Added: Connect to DB, Get all Past history and summarize improvements and Suggest next exercises."


# # Step 1: Get a paragraph from LLM
# if st.button("Test my speech"):
#     client = LLMClient(model="llama2")
#     prompt = "You are a speech therapist agent that can analyses users speech and suggest exercises. First give the user a complicated paragraph to read, it should have 'h' 'm' words. Keep it around 60 words."
#     paragraph = client.generate(prompt)
#     st.session_state.paragraph = paragraph

# if st.button("Analyse my progress"):
#     # client = LLMClient(model="llama2")
#     # prompt = "You are a speech therapist agent that can analyses users speech and suggest exercises. First give the user a complicated paragraph to read, it should have 'h' 'm' words. Keep it around 60 words."
#     # paragraph = client.generate(prompt)
#     st.session_state.paragraph = "To Be Added: Connect to DB, Get all Past history and summarize improvements and Suggest next exercises."
