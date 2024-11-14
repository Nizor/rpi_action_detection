import wave
import json
import subprocess
from vosk import Model, KaldiRecognizer

# Path to your Vosk model directory
MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # Update this path if necessary

# Function to record audio and save it as a WAV file
def record_audio(filename="test.wav", duration=5):
    try:
        # Record audio using arecord
        subprocess.run([
            'arecord', '-D', 'hw:3,0', '-f', 'S16_LE', '-c', '1', '-r', '44100', '-t', 'wav', '-d', str(duration), filename
        ])
    except Exception as e:
        print(f"Error recording audio: {e}")

# Function to transcribe audio using Vosk
def transcribe_audio(filename="test.wav"):
    # Load Vosk model
    model = Model(MODEL_PATH)
    
    # Open the audio file
    with wave.open(filename, "rb") as wf:
        # Check if audio format is compatible with Vosk
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 44100:
            raise ValueError("Audio format not supported. Please record in mono, 16-bit, 44100 Hz.")

        # Initialize recognizer
        recognizer = KaldiRecognizer(model, wf.getframerate())
        
        transcription = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                transcription += result.get("text", "")
        
        # Get the final part of the transcription
        final_result = json.loads(recognizer.FinalResult())
        transcription += final_result.get("text", "")
        
    return transcription

# Test the functions
if __name__ == "__main__":
    audio_file = "test.wav"
    print("Recording audio...")
    record_audio(audio_file, duration=5)
    print("Transcribing audio...")
    text = transcribe_audio(audio_file)
    print("Transcription:", text)
