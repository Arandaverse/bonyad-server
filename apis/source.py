from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from getClosestQuestion import router as get_closest_question

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",  # React app origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.router.include_router(get_closest_question)
