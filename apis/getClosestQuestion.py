from fastapi import APIRouter, File, UploadFile, HTTPException
import uuid
import subprocess
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from ai import inference

Audio_Dir = '../Public/'

router = APIRouter()


# def reencode_audio(input_path: str, output_path: str):
#     command = f"ffmpeg -i {input_path} -ar 16000 {output_path}"
#     subprocess.run(command, shell=True, check=True)


@router.post('/get-closest-question')
async def get_closest_question(file: UploadFile = File(...)):
    if "audio" not in file.content_type:
        raise HTTPException(status_code=400, detail="File must be an audio")

    original_filename = f"{str(uuid.uuid4())[:6]}_original.mp3"
    reencoded_filename = f"{str(uuid.uuid4())[:6]}_reencoded.mp3"
    original_path = os.path.join(Audio_Dir, original_filename)
    reencoded_path = os.path.join(Audio_Dir, reencoded_filename)

    contents = await file.read()

    # Save the original file
    with open(original_path, "wb") as f:
        f.write(contents)

    print(f"Saved original file to: {original_path}")  # Debug log

    # # Ensure the original file is saved correctly
    # if not os.path.isfile(original_path):
    #     raise HTTPException(status_code=500, detail="Original file was not saved correctly")
    #
    # # Re-encode the audio file to ensure it is properly formatted
    # try:
    #     reencode_audio(original_path, reencoded_path)
    # except subprocess.CalledProcessError as e:
    #     raise HTTPException(status_code=500, detail=f"Error re-encoding file: {str(e)}")
    #
    # print(f"Re-encoded file saved to: {reencoded_path}")  # Debug log

    # Ensure the re-encoded file is saved correctly
    # if not os.path.isfile(original_path):
    #     raise HTTPException(status_code=500, detail="Re-encoded file was not saved correctly")
    result = inference.inference(audio_paths=[original_path])

    return result
