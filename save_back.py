from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # temporarily allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase setup
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key:
    raise Exception("SUPABASE_URL and SUPABASE_KEY must be set in .env")

supabase: Client = create_client(url, key)

# User schema
class User(BaseModel):
    email: str
    name: str
    picture: str

# Save user route
@app.post("/save_user")
def save_user(user: User):
    try:
        # Check if already exists
        existing = supabase.table("users").select("*").eq("email", user.email).execute()
        if existing.data:
            return {"message": "User already exists"}

        # Insert user
        result = supabase.table("users").insert({
            "email": user.email,
            "name": user.name,
            "picture": user.picture
        }).execute()

        return {"message": "User saved", "data": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
