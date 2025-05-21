# English Accent Detector

A tool for analyzing spoken English accents from video content for hiring purposes.

## Features

- Accepts public video URLs (YouTube, Loom, Vimeo, or direct MP4 links)
- Extracts audio from videos
- Transcribes speech using OpenAI's Whisper model
- Analyzes accent patterns to identify English variants (American, British, Australian, Indian)
- Provides confidence scoring and detailed explanation
- Simple UI for easy usage

## Demo

You can try the live demo deployed on Streamlit Cloud: [Accent Detector App](https://accent-detector.streamlit.app)

## How It Works

1. **Video Processing**: The app uses `yt-dlp` to download and extract audio from video links
2. **Speech Recognition**: OpenAI's Whisper model converts speech to text
3. **Accent Analysis**: A combination of linguistic and acoustic features are used to classify the accent
4. **Result Visualization**: Clean UI displays accent classification, confidence score, and explanation

## Technical Implementation

This solution employs several techniques:

- **Vocabulary Pattern Matching**: Identifies region-specific words and phrases
- **Acoustic Feature Analysis**: Examines speech rhythm, intonation, and tempo
- **Confidence Scoring**: Calculated based on feature differentiation

## Installation

### Requirements

- Python 3.8+
- FFmpeg (for audio processing)

### Setup

1. Clone this repository
   ```
   git clone https://github.com/yourusername/accent-detector.git
   cd accent-detector
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Install FFmpeg (if not already installed)
   - **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)

## Usage

1. Run the Streamlit app
   ```
   streamlit run accent_detector.py
   ```

2. Open your browser and go to `http://localhost:8501`

3. Enter a video URL and click "Analyze Accent"

## Limitations

- This proof-of-concept uses simplified linguistic patterns
- For production use, a more sophisticated model trained on labeled accent data would be recommended
- Performance may vary with audio quality and speech clarity

## Future Improvements

- Implement a pre-trained neural network for more accurate classification
- Add support for more accent varieties
- Include detailed phonological analysis
- Improve confidence scoring with more sophisticated algorithms

## License

MIT

---

Created by [Your Name] for REM Waste technical challenge
