# ============================================
# PlantGuard AI - Smart Farming Assistant
# Configuration File
# ============================================

import os

# OpenRouter API Configuration (AI Crop Doctor)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "tngtech/deepseek-r1t-chimera:free")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost:5000")
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "PlantGuard AI")

# Weather API Configuration (OpenWeatherMap)
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")

# ElevenLabs Text-to-Speech API Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel

# Application Settings
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() in ("1", "true", "yes")
DEFAULT_LOCATION = os.getenv("DEFAULT_LOCATION", "karachi, PK")

# API Endpoints
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
WEATHER_ENDPOINT = "https://api.openweathermap.org/data/2.5/weather"
WEATHER_FORECAST_ENDPOINT = "https://api.openweathermap.org/data/2.5/forecast"
ELEVENLABS_TTS_ENDPOINT = "https://api.elevenlabs.io/v1/text-to-speech"
