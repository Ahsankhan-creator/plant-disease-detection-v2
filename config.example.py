# Copy values into environment variables (recommended for GitHub/Vercel)
# Do NOT commit real keys.

import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "tngtech/deepseek-r1t-chimera:free")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost:5000")
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "PlantGuard AI")

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# Hugging Face Inference API (recommended for Vercel disease detection)
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mesabo/agri-plant-disease-resnet50")
ENABLE_LOCAL_DISEASE_MODEL = os.getenv("ENABLE_LOCAL_DISEASE_MODEL", "false")

DEBUG_MODE = os.getenv("DEBUG_MODE", "true")
DEFAULT_LOCATION = os.getenv("DEFAULT_LOCATION", "Karachi, PK")
