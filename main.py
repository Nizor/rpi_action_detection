from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
from video_feed import VideoCamera, gen_frames
import tensorflow as tf
from transcription import transcribe_audio
import subprocess
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

model = tf.keras.models.load_model("models/actionModel.keras")

@app.get("/", response_class=HTMLResponse)
async def splash_screen(request: Request):
    return templates.TemplateResponse("splash.html", {"request": request})

@app.get("/menu", response_class=HTMLResponse)
async def menu_page(request: Request):
    return templates.TemplateResponse("menu.html", {"request": request})

@app.get("/voice", response_class=HTMLResponse)
async def voice_page(request: Request):
    return templates.TemplateResponse("voice.html", {"request": request})

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Save the uploaded audio file temporarily
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(file.file.read())
    
    # Convert to a compatible WAV format (PCM, mono, 16-bit, 44100 Hz) using ffmpeg
    converted_file_path = "converted_audio.wav"
    try:
        subprocess.run([
            "ffmpeg", "-i", temp_file_path, "-ac", "1", "-ar", "44100", "-f", "wav", converted_file_path
        ], check=True)
        
        # Transcribe the converted audio file
        transcription = transcribe_audio(converted_file_path)
    except subprocess.CalledProcessError:
        return {"text": "Failed to convert audio file. Ensure it is in a supported format."}
    finally:
        # Clean up temporary files
        os.remove(temp_file_path)
        if os.path.exists(converted_file_path):
            os.remove(converted_file_path)
    
    return {"text": transcription}


@app.get("/home", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(VideoCamera(), model), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0", port=8443, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
