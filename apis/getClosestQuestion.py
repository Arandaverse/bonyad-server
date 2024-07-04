from fastapi import APIRouter, File, UploadFile, HTTPException
import uuid
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from ai import inference

Audio_Dir = '../Public/'

router = APIRouter()


@router.post('/get-closest-question')
async def get_closest_question(file: UploadFile = File(...)):
    if "audio" not in file.content_type:
        raise HTTPException(400, detail="File must be an audio")

    file.filename = f"{str(uuid.uuid4())[:6]}.mp3"
    contents = await file.read()

    # save the file
    audio_path = f"{Audio_Dir}{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(contents)
    result = inference.inference([audio_path])
    os.remove(audio_path)

    return result
