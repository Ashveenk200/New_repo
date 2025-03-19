import streamlit as st
import whisper
import tempfile
import os
import time
import torch
import pandas as pd
from pydub import AudioSegment
from pyannote.audio.pipelines import SpeakerDiarization
from speechbrain.inference import SpeakerRecognition

from huggingface_hub import login
import concurrent.futures
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import glob

# Load SpaCy model for name extraction
nlp = spacy.load("en_core_web_sm")

# ========================== 1. Authentication ==========================
HF_TOKEN = "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"
login(HF_TOKEN)

# ========================== 2. Load Models ==========================
@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("medium", device=device)
    return model, device

model, device = load_whisper_model()

@st.cache_resource
def load_diarization_model():
    try:
        pipeline = SpeakerDiarization.from_pretrained(
            "collinbarnwell/pyannote-speaker-diarization-31", use_auth_token=HF_TOKEN
        )
        pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return pipeline
    except Exception as e:
        st.error(f"Error loading diarization model: {str(e)}")
        return None

diarization_pipeline = load_diarization_model()

# ========================== 3. Streamlit UI ==========================
st.title("🎤 VM")
option = st.radio("Choose an option:", ("Upload Multiple Files", "Select Folder"))

uploaded_files = []

if option == "Upload Multiple Files":
    uploaded_files = st.file_uploader("📂 Upload multiple audio files", type=["mp3", "wav", "m4a"], accept_multiple_files=True)
elif option == "Select Folder":
    folder_path = st.text_input("📂 Enter folder path containing audio files")
    if folder_path:
        uploaded_files = [open(f, "rb") for f in glob.glob(os.path.join(folder_path, "*.wav"))]

# Submit button
if not uploaded_files:
    st.warning("Please upload files or select a folder.")
else:
    if st.button("Submit"):
        data_records = []

        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio_path = temp_audio.name

            try:
                audio = AudioSegment.from_file(file)
                audio.export(temp_audio_path, format="wav")
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                continue

            duration_sec = len(audio) / 1000
            duration_min = duration_sec / 60

            st.audio(file, format=file.type)
            st.write(f"🎵 **Processing {file.name} - Duration:** {int(duration_min)} min {int(duration_sec % 60)} sec")

            st.write("🔄 **Transcribing and Identifying Speakers... Please wait.**")
            start_time = time.time()

            try:
                if diarization_pipeline is None:
                    st.error("Speaker diarization model could not be loaded.")
                    os.remove(temp_audio_path)
                    continue

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    diarization_future = executor.submit(diarization_pipeline, temp_audio_path)
                    whisper_future = executor.submit(lambda: model.transcribe(temp_audio_path, fp16=(device == "cuda"), temperature=0.0, word_timestamps=True))

                    diarization_result = diarization_future.result()
                    whisper_result = whisper_future.result()

                transcribed_text = whisper_result.get("segments", [])
                if not transcribed_text:
                    st.error(f"No speech detected in {file.name}. Please upload a clearer file.")
                    os.remove(temp_audio_path)
                    continue

                speaker_segments = [(turn.start, turn.end, speaker) for turn, _, speaker in diarization_result.itertracks(yield_label=True)]
                speaker_names = {}
                speaker_transcriptions = {}
                full_transcript = ""

                for segment in transcribed_text:
                    segment_start = segment['start']
                    segment_text = segment['text']
                    full_transcript += " " + segment_text

                    for start, end, speaker in speaker_segments:
                        if start <= segment_start <= end:
                            if speaker not in speaker_names:
                                speaker_names[speaker] = f"Speaker {len(speaker_names) + 1}"
                            speaker_name = speaker_names[speaker]
                            speaker_transcriptions.setdefault(speaker_name, []).append(segment_text)

                doc = nlp(full_transcript)
                extracted_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
                analyzer = SentimentIntensityAnalyzer()
                sentiment_results = {}

                for speaker, texts in speaker_transcriptions.items():
                    full_text = " ".join(texts)
                    sentiment = analyzer.polarity_scores(full_text)
                    sentiment_results[speaker] = {
                        "pos": round(sentiment['pos'] * 100, 2),
                        "neu": round(sentiment['neu'] * 100, 2),
                        "engagement": round((sentiment['pos'] - sentiment['neu']) * 100, 2),
                        "interest": "Interested" if sentiment['pos'] > 30 and sentiment['neu'] < 60 else "Not Interested" if sentiment['pos'] < 25 and sentiment['neu'] > 70 else "Moderately Interested"
                    }

                end_time = time.time()
                processing_time = round((end_time - start_time) / 60, 2)

                record = [
                    len(data_records) + 1, file.name, " ".join(speaker_transcriptions.get("Speaker 1", [])),
                    " ".join(speaker_transcriptions.get("Speaker 2", [])), time.strftime('%H:%M:%S', time.localtime(start_time)),
                    time.strftime('%H:%M:%S', time.localtime(end_time)), round(duration_min, 2), processing_time,
                    sentiment_results.get("Speaker 1", {}).get("pos", "-"),
                    sentiment_results.get("Speaker 1", {}).get("neu", "-"),
                    sentiment_results.get("Speaker 1", {}).get("engagement", "-"),
                    sentiment_results.get("Speaker 2", {}).get("pos", "-"),
                    sentiment_results.get("Speaker 2", {}).get("neu", "-"),
                    sentiment_results.get("Speaker 2", {}).get("engagement", "-"),
                    sentiment_results.get("Speaker 1", {}).get("interest", "-")
                ]
                data_records.append(record)

                # Display DataFrame Incrementally
                df = pd.DataFrame(data_records, columns=["Sno.", "File Name", "Person 1 Chat", "Person 2 Chat", "Time Start", "Time End", "Audio Duration (min)", "Total Processing Time (min)", "Person 1 Positive Score", "Person 1 Neutral Score", "Person 1 Engagement Score", "Person 2 Positive Score", "Person 2 Neutral Score", "Person 2 Engagement Score", "Interest Level"])
                st.dataframe(df)

            except Exception as e:
                st.error(f"Error during processing {file.name}: {str(e)}")
            os.remove(temp_audio_path)
