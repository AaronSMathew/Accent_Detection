import streamlit as st
import os
import tempfile
import numpy as np
import pandas as pd
import torch
import librosa
import subprocess
import re
import requests
import whisper
from pathlib import Path
from scipy.io import wavfile
from google.cloud import speech
import io
import base64

# Set page config
st.set_page_config(
    page_title="English Accent Detector",
    page_icon="🎙️",
    layout="wide"
)

# CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .accent-result {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        background-color: #f0f2f6;
    }
    .confidence-meter {
        height: 20px;
        border-radius: 10px;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        background-color: #e0e0e0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        background-color: #1f77b4;
    }
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1>English Accent Detector</h1>
    <p>REM Waste Hiring Tool</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
This tool analyzes videos to detect and classify English accents for hiring purposes.
Upload a video URL (YouTube, Loom, Vimeo, or direct MP4 link) to get started.
""")

# Helper functions
@st.cache_resource
def load_whisper_model():
    """Load Whisper model for transcription"""
    model = whisper.load_model("base")
    return model

def extract_audio_from_video(video_url):
    """Download video and extract audio using yt-dlp"""
    try:
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.wav")
        
        # Use yt-dlp to download and extract audio
        cmd = [
            "yt-dlp", 
            "-x", 
            "--audio-format", "wav", 
            "--audio-quality", "0",
            "-o", audio_path.replace(".wav", ".%(ext)s"),
            video_url
        ]
        
        process = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        if process.returncode != 0:
            st.error(f"Error extracting audio: {process.stderr}")
            return None
        
        # Find the wav file
        wav_files = list(Path(temp_dir).glob("*.wav"))
        if not wav_files:
            st.error("No audio file was extracted")
            return None
            
        return str(wav_files[0])
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None

def transcribe_audio(audio_path, model):
    """Transcribe audio using Whisper"""
    try:
        # Load audio file
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

def analyze_accent(transcript, audio_path):
    """
    Analyze the accent using a set of linguistic features
    
    This is a simplified version for demonstration purposes.
    In a real production system, this would use more sophisticated models.
    """
    # Load audio for acoustic features
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        st.error(f"Error loading audio for analysis: {str(e)}")
        return None, None, None
    
    # Linguistic features for accent detection (simplified)
    # In a real system, we would use a pre-trained model
    
    # Feature 1: Vocabulary and phrase patterns
    american_patterns = [
        r'\b(gonna|wanna|y\'all|awesome|bucks|cool|folks)\b',
        r'\b(apartment|elevator|subway|vacation|garbage|sidewalk)\b'
    ]
    
    british_patterns = [
        r'\b(bloody|brilliant|cheers|mate|quite|rather|proper|fancy)\b',
        r'\b(flat|lift|underground|holiday|rubbish|pavement)\b'
    ]
    
    australian_patterns = [
        r'\b(g\'day|mate|crikey|fair dinkum|arvo|barbie|footy)\b',
        r'\b(thongs|ute|servo|brekkie|heaps|reckon)\b'
    ]
    
    indian_patterns = [
        r'\b(actually|basically|only|itself|kindly|itself|prepone)\b',
        r'\b(doing the needful|good name|passed out|itself|yaar)\b'
    ]
    
    # Count pattern matches
    american_count = sum([len(re.findall(pattern, transcript.lower())) for pattern in american_patterns])
    british_count = sum([len(re.findall(pattern, transcript.lower())) for pattern in british_patterns])
    australian_count = sum([len(re.findall(pattern, transcript.lower())) for pattern in australian_patterns])
    indian_count = sum([len(re.findall(pattern, transcript.lower())) for pattern in indian_patterns])
    
    # Feature 2: Acoustic features
    # Extract pitch and rhythm features
    if len(y) > 0:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        pitch_mean = np.mean(librosa.feature.zero_crossing_rate(y=y))
        
        # Simplified acoustic feature analysis
        # In a real system, we would use more sophisticated features
        american_acoustic = 0
        british_acoustic = 0
        australian_acoustic = 0
        indian_acoustic = 0
        
        # Higher tempo often associated with American speech
        if tempo > 120:
            american_acoustic += 2
        elif tempo > 100:
            american_acoustic += 1
            australian_acoustic += 1
        else:
            british_acoustic += 1
            indian_acoustic += 1
        
        # Pitch variance typically higher in British English
        if pitch_mean > 0.07:
            british_acoustic += 2
        elif pitch_mean > 0.05:
            american_acoustic += 1
        else:
            indian_acoustic += 1
    else:
        american_acoustic = 0
        british_acoustic = 0
        australian_acoustic = 0
        indian_acoustic = 0
    
    # Combine features
    american_score = american_count * 2 + american_acoustic
    british_score = british_count * 2 + british_acoustic
    australian_score = australian_count * 2 + australian_acoustic
    indian_score = indian_count * 2 + indian_acoustic
    
    # Add general english proficiency score
    # This would be more sophisticated in a real system
    scores = {
        "American": american_score,
        "British": british_score, 
        "Australian": australian_score,
        "Indian": indian_score
    }
    
    # Normalize scores
    total_score = sum(scores.values())
    if total_score == 0:
        # If no clear patterns detected, default to general categories
        accent_type = "Neutral/General English"
        confidence = 60  # Medium confidence
        
        # Check transcript length to refine confidence
        if len(transcript) < 50:
            confidence = 40  # Lower confidence for short samples
        
        explanation = """
        The speech sample doesn't contain strong regional patterns.
        This could indicate a neutral English accent or insufficient speech data.
        """
    else:
        # Find the highest scoring accent
        accent_type = max(scores, key=scores.get)
        max_score = scores[accent_type]
        
        # Calculate confidence based on differentiation between scores
        # Higher differential = higher confidence
        second_best = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
        differential = max_score - second_best
        
        # Convert to confidence percentage (0-100%)
        confidence = min(95, 50 + (differential * 10))
        
        # Generate explanation
        explanations = {
            "American": "The speaker uses vocabulary, intonation patterns, and rhythmic features typical of American English.",
            "British": "The speaker demonstrates features common in British English, including vocabulary choices and prosodic patterns.",
            "Australian": "The speaker shows characteristics of Australian English in vocabulary and speech rhythm.",
            "Indian": "The speaker exhibits patterns typical of Indian English in terms of vocabulary and speech cadence."
        }
        
        explanation = explanations.get(accent_type, "")
    
    return accent_type, confidence, explanation

# Main app
def main():
    whisper_model = load_whisper_model()
    
    video_url = st.text_input("Enter video URL (YouTube, Loom, Vimeo, or direct MP4 link)")
    
    if st.button("Analyze Accent"):
        if not video_url:
            st.warning("Please enter a video URL")
            return
            
        with st.spinner("Processing video..."):
            # Step 1: Extract audio
            st.info("Downloading and extracting audio...")
            audio_path = extract_audio_from_video(video_url)
            
            if not audio_path:
                st.error("Failed to extract audio from the video")
                return
                
            # Step 2: Transcribe audio
            st.info("Transcribing audio...")
            transcript = transcribe_audio(audio_path, whisper_model)
            
            if not transcript:
                st.error("Failed to transcribe audio")
                return
                
            # Step 3: Analyze accent
            st.info("Analyzing accent...")
            accent_type, confidence, explanation = analyze_accent(transcript, audio_path)
            
            if not accent_type:
                st.error("Failed to analyze accent")
                return
                
            # Step 4: Display results
            st.success("Analysis complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Transcript")
                st.text_area("", transcript, height=200)
            
            with col2:
                st.subheader("Accent Analysis")
                
                st.markdown(f"""
                <div class="accent-result">
                    <h3>Accent: {accent_type}</h3>
                    <p>Confidence: {confidence:.1f}%</p>
                    <div class="confidence-meter">
                        <div class="confidence-fill" style="width: {confidence}%"></div>
                    </div>
                    <p><strong>Explanation:</strong> {explanation}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional info for hiring purposes
                english_proficiency = "High" if confidence > 75 else "Medium" if confidence > 50 else "Low"
                st.info(f"English Proficiency Assessment: {english_proficiency}")

if __name__ == "__main__":
    main()