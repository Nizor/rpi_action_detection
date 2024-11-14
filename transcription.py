from vosk import Model, KaldiRecognizer
import wave
import json
import os

# Define the path to your Vosk model
MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # Update this path if necessary

# Function to transcribe audio using Vosk
def transcribe_audio(file_path: str) -> str:
    """Transcribe audio file using Vosk and return the transcription text."""
    
    # Check if the model directory exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Vosk model not found at path: {MODEL_PATH}")

    # Load the Vosk model
    model = Model(MODEL_PATH)
    transcription = ""

    # Open the audio file
    with wave.open(file_path, "rb") as wf:
        # Check for required audio format (mono, 16-bit, 44100 Hz)
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 44100:
            raise ValueError("Audio format not supported. Please record in mono, 16-bit, 44100 Hz.")

        # Initialize recognizer
        recognizer = KaldiRecognizer(model, wf.getframerate())

        # Process each audio frame and accumulate results
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                transcription += result.get("text", "")

        # Append final result
        final_result = json.loads(recognizer.FinalResult())
        transcription += final_result.get("text", "")

    return transcription
