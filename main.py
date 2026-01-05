# ============================================
# PlantGuard AI - Smart Farming Assistant
# FastAPI Backend
# ============================================

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import httpx
import io
import json
from PIL import Image, UnidentifiedImageError
import re
import asyncio
import random
import os

# Import configuration
from config import (
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_ENDPOINT,
    OPENROUTER_SITE_URL, OPENROUTER_SITE_NAME,
    WEATHER_API_KEY, WEATHER_ENDPOINT, WEATHER_FORECAST_ENDPOINT,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_TTS_ENDPOINT,
    DEFAULT_LOCATION, DEBUG_MODE
)

# Initialize FastAPI app
app = FastAPI(
    title="PlantGuard AI - Smart Farming Assistant",
    description="AI-Powered Smart Irrigation, Weather Prediction, Pest & Disease Detection, and Voice-Enabled Crop Doctor",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================
# Plant Disease Detection Backend
# ============================================
# Normal (local) setup:
# - Uses a local transformers image-classification model by default.
# - No external API token is required for disease detection.
# - If you want to disable local loading (e.g., limited environment), set ENABLE_LOCAL_DISEASE_MODEL=false.

ENABLE_LOCAL_DISEASE_MODEL = os.getenv("ENABLE_LOCAL_DISEASE_MODEL", "true").lower() in ("1", "true", "yes")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mesabo/agri-plant-disease-resnet50")

disease_model = None
disease_processor = None
MODEL_LOADED = False
DISEASE_BACKEND = "local" if ENABLE_LOCAL_DISEASE_MODEL else "none"

if ENABLE_LOCAL_DISEASE_MODEL:
    print("🌱 Loading local Plant Disease Detection Model...")
    try:
        import torch  # type: ignore
        from transformers import AutoModelForImageClassification, AutoImageProcessor  # type: ignore

        disease_model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
        disease_processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
        disease_model.eval()
        MODEL_LOADED = True
        DISEASE_BACKEND = "local"
        print("✅ Local model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Local model loading failed: {e}")
        MODEL_LOADED = False
        disease_model = None
        disease_processor = None
        DISEASE_BACKEND = "none"

# ============================================
# Pydantic Models
# ============================================
class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None

class WeatherRequest(BaseModel):
    location: str = DEFAULT_LOCATION

class TTSRequest(BaseModel):
    text: str

class IrrigationRequest(BaseModel):
    crop_type: str
    soil_moisture: float
    temperature: float
    humidity: float
    last_watered_hours: int

class DiseaseAdviceRequest(BaseModel):
    disease: str
    confidence: float
    is_healthy: bool
    top_predictions: Optional[list] = None

# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    """Open the UI by default."""
    return RedirectResponse(url="/static/index.html")


@app.get("/api/health")
async def health():
    """Health check"""
    return {
        "status": "online",
        "app": "PlantGuard AI - Smart Farming Assistant",
        "features": [
            "Weather Prediction",
            "Plant Disease Detection",
            "AI Crop Doctor Chat",
            "Voice Assistant (TTS)",
            "Smart Irrigation Recommendations",
        ],
        "model_loaded": MODEL_LOADED,
        "disease_backend": DISEASE_BACKEND,
    }

# ============================================
# Weather API Endpoints
# ============================================

@app.post("/api/weather")
async def get_weather(request: WeatherRequest):
    """Get current weather data for a location"""
    if not WEATHER_API_KEY:
        raise HTTPException(status_code=503, detail="WEATHER_API_KEY is not configured")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                WEATHER_ENDPOINT,
                params={
                    "q": request.location,
                    "appid": WEATHER_API_KEY,
                    "units": "metric"
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Weather API error")
            
            data = response.json()
            
            return {
                "status": "success",
                "location": data.get("name", request.location),
                "country": data.get("sys", {}).get("country", ""),
                "temperature": data.get("main", {}).get("temp"),
                "feels_like": data.get("main", {}).get("feels_like"),
                "humidity": data.get("main", {}).get("humidity"),
                "pressure": data.get("main", {}).get("pressure"),
                "wind_speed": data.get("wind", {}).get("speed"),
                "description": data.get("weather", [{}])[0].get("description", ""),
                "icon": data.get("weather", [{}])[0].get("icon", ""),
                "visibility": data.get("visibility", 0) / 1000,  # Convert to km
                "clouds": data.get("clouds", {}).get("all", 0)
            }
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Weather service unavailable: {str(e)}")

@app.post("/api/weather/forecast")
async def get_weather_forecast(request: WeatherRequest):
    """Get 5-day weather forecast for a location"""
    if not WEATHER_API_KEY:
        raise HTTPException(status_code=503, detail="WEATHER_API_KEY is not configured")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                WEATHER_FORECAST_ENDPOINT,
                params={
                    "q": request.location,
                    "appid": WEATHER_API_KEY,
                    "units": "metric"
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Weather API error")
            
            data = response.json()
            
            # Process forecast data
            forecasts = []
            for item in data.get("list", [])[:10]:  # Get first 10 forecasts (30+ hours)
                forecasts.append({
                    "datetime": item.get("dt_txt"),
                    "temperature": item.get("main", {}).get("temp"),
                    "humidity": item.get("main", {}).get("humidity"),
                    "description": item.get("weather", [{}])[0].get("description", ""),
                    "icon": item.get("weather", [{}])[0].get("icon", ""),
                    "wind_speed": item.get("wind", {}).get("speed"),
                    "rain_probability": item.get("pop", 0) * 100
                })
            
            return {
                "status": "success",
                "location": data.get("city", {}).get("name", request.location),
                "country": data.get("city", {}).get("country", ""),
                "forecasts": forecasts
            }
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Weather service unavailable: {str(e)}")

# ============================================
# Plant Disease Detection Endpoint
# ============================================

@app.post("/api/detect-disease")
async def detect_disease(
    file: UploadFile = File(...),
    with_ai_advice: bool = Query(True, description="If true, also returns AI-generated advice")
):
    """Detect plant disease from uploaded image"""
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Disease detection is not configured. Enable local model loading (ENABLE_LOCAL_DISEASE_MODEL=true) and install torch/transformers.",
        )
    
    try:
        allowed_ext = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        is_image_by_mime = bool(file.content_type) and file.content_type.startswith("image/")
        is_image_by_ext = bool(file.filename) and file.filename.lower().endswith(allowed_ext)
        if not (is_image_by_mime or is_image_by_ext):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a valid image (JPG/PNG/WEBP).")

        # Read and process image
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Validate image is readable
        Image.open(io.BytesIO(image_data)).convert("RGB")

        # Predict using the local model
        if disease_model is None or disease_processor is None:
            raise HTTPException(status_code=503, detail="Disease detection model is not loaded")

        import torch  # type: ignore

        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        inputs = disease_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = disease_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top_probs, top_indices = torch.topk(probs[0], 3)

            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                predictions.append(
                    {
                        "disease": disease_model.config.id2label[idx.item()],
                        "confidence": round(prob.item() * 100, 2),
                    }
                )

        if not predictions:
            raise HTTPException(status_code=502, detail="Disease detection returned no predictions")

        primary_prediction = predictions[0]

        result = {
            "status": "success",
            "disease": primary_prediction["disease"],
            "confidence": primary_prediction["confidence"],
            "top_predictions": predictions,
            "is_healthy": "healthy" in str(primary_prediction["disease"]).lower(),
            "disease_backend": DISEASE_BACKEND,
        }

        if with_ai_advice:
            try:
                advice = await get_ai_disease_advice(
                    DiseaseAdviceRequest(
                        disease=result["disease"],
                        confidence=float(result["confidence"]),
                        is_healthy=bool(result["is_healthy"]),
                        top_predictions=result["top_predictions"],
                    )
                )
                result["ai_advice"] = advice.get("advice")
                result["ai_advice_status"] = advice.get("status")
            except HTTPException as e:
                result["ai_advice"] = None
                result["ai_advice_status"] = "error"
                result["ai_advice_error"] = e.detail
            except Exception as e:
                result["ai_advice"] = None
                result["ai_advice_status"] = "error"
                result["ai_advice_error"] = f"AI advice failed: {str(e)}"

        return result
            
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read image. Please upload a valid JPG/PNG/WEBP.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease detection failed: {str(e)}")


def _require_openrouter_key() -> None:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY is not configured")


def _user_greeted(user_text: str) -> bool:
    if not user_text:
        return False
    return bool(re.match(r"^\s*(hi|hello|hey|greetings|assalam\s*o\s*alaikum|salam|aoa)\b", user_text, re.IGNORECASE))


def _strip_greetings(text: str, *, user_message: str) -> str:
    """Remove unsolicited greetings from the AI.

    If the user greeted first, keep greetings (user wants a friendly reply).
    """
    if not text:
        return ""
    cleaned = text.strip()

    if _user_greeted(user_message):
        return cleaned

    greeting_re = re.compile(r"^(hi|hello|hey|greetings|assalam\s*o\s*alaikum|salam|aoa)\b[!,\.\s]*", re.IGNORECASE)
    for _ in range(3):
        lines = cleaned.splitlines()
        if not lines:
            break
        first = lines[0].strip()
        first2 = greeting_re.sub("", first).strip()
        if first2 != first:
            lines[0] = first2
            cleaned = "\n".join([ln for ln in lines if ln.strip() != ""]).strip()
        else:
            break

    return cleaned.strip()


def _clean_ai_response(text: str, user_message: str) -> str:
    """Clean AI response to keep it human-like and remove model/meta talk.

    This intentionally avoids aggressive rewriting; it mainly removes:
    - unsolicited greetings (when user didn't greet)
    - "according to the model" / confidence / analysis prefaces
    """
    if not text:
        return ""

    cleaned = _strip_greetings(text.strip(), user_message=user_message)

    user_used_emoji = bool(re.search(r"[\U0001F300-\U0001FAFF]", user_message or ""))

    # Remove common "meta" prefixes from the beginning
    leading_patterns = [
        r"according to (?:the )?(?:model|analysis|prediction)",
        r"based on (?:the )?(?:image|model|prediction)",
        r"the (?:model|system) (?:predicts|says|indicates)",
        r"here(?:'s| is) (?:the |an )?analysis",
        r"to summarize",
        r"in summary",
        r"in conclusion",
        r"as an ai",
        r"as a (?:model|assistant)",
    ]
    for phrase in leading_patterns:
        cleaned = re.sub(fr"^\s*{phrase}[,\s:.-]*", "", cleaned, flags=re.IGNORECASE).strip()

    # Remove lines that are clearly meta-analysis (keep practical bullets)
    meta_indicators = [
        "according to",
        "the model",
        "confidence",
        "probability",
        "analysis",
        "prediction",
        "dataset",
        "hugging face",
    ]

    # Remove internal-monologue / chain-of-thought style lines
    internal_thought_indicators = [
        "the user",
        "user greeted",
        "i need to",
        "i should",
        "let's keep",
        "let me think",
        "i'm going to",
        "my goal is",
        "step by step",
        "reasoning",
        "chain of thought",
        "as an ai",
        "language model",
        "i'm just a bunch of code",
        "i can't have feelings",
    ]
    filtered_lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower = line.lower()
        if any(tok in lower for tok in internal_thought_indicators):
            continue

        if any(tok in lower for tok in meta_indicators):
            # If it looks like an action list item, keep it (e.g., "- remove infected leaves")
            if re.match(r"^(\-|\*|\u2022|\d+[\.)])\s+", line):
                filtered_lines.append(line)
            continue

        filtered_lines.append(line)

    cleaned = "\n".join(filtered_lines).strip()

    # Remove emojis unless the user used emojis first
    if not user_used_emoji and cleaned:
        cleaned = re.sub(r"[\U0001F300-\U0001FAFF]", "", cleaned).strip()

    # EMERGENCY FILTER: If it still sounds like thinking, rewrite the opening
    thinking_patterns = [
        r"okay,?\s+(?:the\s+)?user",
        r"alright,?\s+(?:the\s+)?user",
        r"the\s+user\s+greet",
        r"first,?\s+(?:i\s+)?need\s+to",
        r"let\s+me\s+(?:explain|describe|break\s+down)",
        r"i\s+should\s+(?:mention|explain|describe)",
        r"here's?\s+what\s+i\s+think",
        r"starting\s+with",
        r"moving\s+on\s+to",
    ]

    first_paragraph = cleaned.split("\n")[0] if "\n" in cleaned else cleaned
    for pattern in thinking_patterns:
        if re.search(pattern, first_paragraph[:200], re.IGNORECASE):
            # It's still thinking! Force a rewrite.
            tail = cleaned
            if "." in cleaned:
                tail = cleaned[cleaned.find(".") + 1 :].strip() or cleaned

            if "prevent" in user_message.lower() or "how to" in user_message.lower():
                return ("Here’s how to prevent that issue: " + tail).strip()
            return ("Here’s what you can do: " + tail).strip()

    return cleaned


def _is_complex_query(user_message: str) -> bool:
    """Heuristic: longer/multi-part questions feel more natural with a short pause."""
    if not user_message:
        return False
    msg = user_message.strip()
    if _user_greeted(msg) and len(msg) <= 12:
        return False
    # Multi-part / detailed questions
    if len(msg) >= 90:
        return True
    if msg.count("?") >= 2:
        return True
    if sum(msg.lower().count(w) for w in ["how", "why", "what", "when", "where", "should", "can i", "help"]) >= 2:
        return True
    return False


async def _maybe_cognitive_pause(user_message: str) -> None:
    """Small, variable delay for complex queries to feel less robotic."""
    if not _is_complex_query(user_message):
        return
    # Keep it short so the app stays responsive.
    await asyncio.sleep(random.uniform(0.25, 0.8))


def _sentiment_hint(user_message: str) -> str:
    """Very small sentiment mirroring hint for the prompt (no heavy NLP)."""
    msg = (user_message or "").lower()
    upset = any(w in msg for w in ["frustrated", "angry", "upset", "not working", "problem", "worried", "stress", "failed"])
    positive = any(w in msg for w in ["great", "awesome", "thanks", "thank you", "good news", "love", "happy"])
    if upset:
        return "User seems frustrated or worried. Respond with calm empathy first (one short line), then help."
    if positive:
        return "User seems positive. Match that energy briefly, then help."
    return ""


# Shared system prompts
GENERAL_SYSTEM_PROMPT = """You are a friendly, human-like agricultural assistant.

Rules:
- Reply naturally like a real person
- If user says "hi", reply with a short greeting (one sentence)
- If user asks a question, explain it simply
- Do NOT list datasets, links, or resources unless user asks
- Do NOT sound like a search engine
- Use short, clear, conversational answers
- Explain things as if talking to a farmer or student

Hard rules (very important):
- Do NOT write internal thoughts like: "Alright, the user..." / "I need to..." / "Let me think step-by-step"
- Do NOT mention you're an AI, a model, or "code"
- Do NOT mention confidence/probabilities unless the user explicitly asks
- Emojis only if the user uses emojis first

Style tips (important):
- Avoid sounding repetitive; vary your phrasing naturally.
- For complex questions, it's okay to add ONE brief thinking-aloud line like: "Give me a second" or "Let me check that" (no analysis).
- If you don't know, be honest in a human way (e.g., "Hmm, I'm not fully sure") and suggest one practical next step.

Small talk:
- If asked "how are you", reply warmly in 1–2 sentences, then ask what they need help with.

If the user asks about plant disease:
- Explain symptoms first
- Then give prevention tips
- Avoid scientific jargon
- Sound supportive and practical
"""


DISEASE_ADVICE_SYSTEM_PROMPT = """You are a practical, experienced farmer giving advice to a friend.

IMPORTANT:
- Speak like a human, not a robot
- Give direct advice without prefacing it
- Use simple language
- If unsure, say "I'm not completely sure" (do not give percentages)
- Focus on what the farmer should DO

Tone:
- Short, practical, and supportive.
- You can use a small thinking-aloud phrase once ("Let me think…") but don't write analysis.

NEVER include:
- "The model predicts" / "According to the model" / "Confidence"
- Datasets, links, or resources (unless the user asks)
- Any internal monologue like "the user asked..." or step-by-step reasoning
- Emojis unless the user used emojis first
"""


async def _openrouter_chat(
    system_prompt: str,
    user_message: str,
    *,
    max_tokens: int = 400,
    temperature: float = 0.2,
    remove_reasoning: bool = True,
) -> str:
    _require_openrouter_key()
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OPENROUTER_ENDPOINT,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": OPENROUTER_SITE_URL,
                "X-Title": OPENROUTER_SITE_NAME,
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": 0.9,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
            },
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"AI service error: {response.text}")

        data = response.json()
        ai_response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if remove_reasoning:
            return _clean_ai_response(ai_response or "", user_message)
        return _strip_greetings(ai_response or "", user_message=user_message)


@app.post("/api/disease-advice")
async def get_ai_disease_advice(request: DiseaseAdviceRequest):
    """Generate treatment/prevention advice from AI using the model's prediction output."""
    try:
        low_confidence = float(request.confidence) < 60.0

        if request.is_healthy:
            user_message = "My plant looks healthy. What should I do to keep it healthy for the next 1–2 weeks?"
        else:
            if low_confidence:
                user_message = (
                    f"I'm not completely sure, but I think my plant may have {request.disease}. "
                    "What should I check for, and what should I do now?"
                )
            else:
                user_message = (
                    f"I think my plant has {request.disease}. "
                    "What should I check for, and what should I do now?"
                )

        advice = await _openrouter_chat(
            DISEASE_ADVICE_SYSTEM_PROMPT,
            user_message,
            max_tokens=600,
            temperature=0.35,
            remove_reasoning=True,
        )
        if not advice.strip():
            raise HTTPException(status_code=502, detail="AI returned an empty response")

        return {"status": "success", "advice": advice, "model": OPENROUTER_MODEL}
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"AI service unavailable: {str(e)}")

# ============================================
# AI Crop Doctor Chat Endpoint
# ============================================

@app.post("/api/chat")
async def chat_with_crop_doctor(request: ChatRequest):
    """Chat with AI Crop Doctor using OpenRouter API"""
    try:
        # Cognitive pause for complex queries (makes responses feel more natural)
        await _maybe_cognitive_pause(request.message)

        system_prompt = GENERAL_SYSTEM_PROMPT
        hint = _sentiment_hint(request.message)
        if hint:
            system_prompt += f"\n\nTone hint: {hint}"
        
        if request.context:
            system_prompt += f"\n\nContext about this conversation: {request.context}"

        ai_response = await _openrouter_chat(
            system_prompt,
            request.message,
            max_tokens=350,
            temperature=0.3,
            remove_reasoning=True,
        )

        return {
            "status": "success",
            "response": ai_response,
            "model": OPENROUTER_MODEL
        }
            
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"AI service unavailable: {str(e)}")

# ============================================
# Text-to-Speech Endpoint
# ============================================

@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using ElevenLabs API"""
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY is not configured")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ELEVENLABS_TTS_ENDPOINT}/{ELEVENLABS_VOICE_ID}",
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg"
                },
                json={
                    "text": request.text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"TTS service error: {response.text}")
            
            # Return audio as streaming response
            return StreamingResponse(
                io.BytesIO(response.content),
                media_type="audio/mpeg",
                headers={"Content-Disposition": "attachment; filename=speech.mp3"}
            )
            
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"TTS service unavailable: {str(e)}")

# ============================================
# Smart Irrigation Recommendation Endpoint
# ============================================

@app.post("/api/irrigation")
async def get_irrigation_recommendation(request: IrrigationRequest):
    """Get smart irrigation recommendations based on conditions"""
    
    # Calculate irrigation need score (0-100)
    irrigation_score = 0
    recommendations = []
    
    # Soil moisture analysis (lower = needs more water)
    if request.soil_moisture < 20:
        irrigation_score += 40
        recommendations.append("⚠️ Critical: Soil moisture is very low. Immediate irrigation required.")
    elif request.soil_moisture < 40:
        irrigation_score += 25
        recommendations.append("🔶 Moderate: Soil is getting dry. Consider watering soon.")
    elif request.soil_moisture < 60:
        irrigation_score += 10
        recommendations.append("✅ Good: Soil moisture is adequate for most crops.")
    else:
        recommendations.append("💧 Excellent: Soil has sufficient moisture. No immediate watering needed.")
    
    # Temperature analysis
    if request.temperature > 35:
        irrigation_score += 20
        recommendations.append("🌡️ High temperature detected. Increase watering frequency.")
    elif request.temperature > 28:
        irrigation_score += 10
        recommendations.append("🌡️ Warm conditions. Monitor soil moisture more frequently.")
    
    # Humidity analysis
    if request.humidity < 30:
        irrigation_score += 15
        recommendations.append("💨 Low humidity. Evaporation rate is high.")
    elif request.humidity < 50:
        irrigation_score += 5
    
    # Time since last watering
    if request.last_watered_hours > 48:
        irrigation_score += 20
        recommendations.append(f"⏰ It's been {request.last_watered_hours} hours since last watering.")
    elif request.last_watered_hours > 24:
        irrigation_score += 10
    
    # Determine action
    if irrigation_score >= 60:
        action = "WATER NOW"
        action_color = "red"
    elif irrigation_score >= 40:
        action = "WATER SOON"
        action_color = "orange"
    elif irrigation_score >= 20:
        action = "MONITOR"
        action_color = "yellow"
    else:
        action = "NO ACTION NEEDED"
        action_color = "green"
    
    # Crop-specific advice
    crop_advice = get_crop_water_advice(request.crop_type)
    
    return {
        "status": "success",
        "irrigation_score": min(irrigation_score, 100),
        "action": action,
        "action_color": action_color,
        "recommendations": recommendations,
        "crop_advice": crop_advice,
        "input_summary": {
            "crop_type": request.crop_type,
            "soil_moisture": request.soil_moisture,
            "temperature": request.temperature,
            "humidity": request.humidity,
            "last_watered_hours": request.last_watered_hours
        }
    }

def get_crop_water_advice(crop_type: str) -> str:
    """Get watering advice for specific crop types"""
    crop_advice = {
        "tomato": "Tomatoes need consistent moisture. Water deeply 1-2 times per week. Avoid wetting leaves.",
        "corn": "Corn requires about 1 inch of water per week. Critical during tasseling and ear development.",
        "wheat": "Wheat needs moderate water. Critical during heading and grain filling stages.",
        "rice": "Rice needs flooded conditions. Maintain 2-4 inches of standing water during growth.",
        "cotton": "Cotton is drought-tolerant but needs water during flowering and boll development.",
        "potato": "Potatoes need consistent moisture. Water when top 1-2 inches of soil is dry.",
        "soybean": "Soybeans need about 0.5 inches of water per day during pod development.",
        "vegetables": "Most vegetables need 1-2 inches of water per week. Water in the morning.",
        "fruits": "Fruit trees need deep watering. Water less frequently but more thoroughly.",
    }
    
    return crop_advice.get(crop_type.lower(), 
        f"For {crop_type}: Generally water when top 2 inches of soil feels dry. Adjust based on growth stage.")

# ============================================
# Combined Analysis Endpoint
# ============================================

@app.post("/api/full-analysis")
async def full_farm_analysis(
    file: UploadFile = File(None),
    location: str = Form(DEFAULT_LOCATION),
    crop_type: str = Form("vegetables"),
    soil_moisture: float = Form(50.0),
    last_watered_hours: int = Form(24)
):
    """Perform comprehensive farm analysis"""
    results = {
        "status": "success",
        "analysis": {}
    }
    
    # Get weather data
    try:
        weather_request = WeatherRequest(location=location)
        weather_data = await get_weather(weather_request)
        results["analysis"]["weather"] = weather_data
    except Exception as e:
        results["analysis"]["weather"] = {"error": str(e)}
    
    # Disease detection if image provided
    if file and file.filename:
        try:
            disease_data = await detect_disease(file)
            results["analysis"]["disease_detection"] = disease_data
        except Exception as e:
            results["analysis"]["disease_detection"] = {"error": str(e)}
    
    # Irrigation recommendation
    try:
        temp = results["analysis"].get("weather", {}).get("temperature", 25)
        humidity = results["analysis"].get("weather", {}).get("humidity", 50)
        
        irrigation_request = IrrigationRequest(
            crop_type=crop_type,
            soil_moisture=soil_moisture,
            temperature=temp,
            humidity=humidity,
            last_watered_hours=last_watered_hours
        )
        irrigation_data = await get_irrigation_recommendation(irrigation_request)
        results["analysis"]["irrigation"] = irrigation_data
    except Exception as e:
        results["analysis"]["irrigation"] = {"error": str(e)}
    
    return results

# ============================================
# Run the application
# ============================================

if __name__ == "__main__":
    import uvicorn
    # NOTE:
    # Uvicorn's `reload` only works when the app is provided as an import string
    # (e.g., `uvicorn main:app --reload`). When calling uvicorn.run(app, ...)
    # directly (like `python main.py`), enabling reload will crash.
    if DEBUG_MODE:
        print("ℹ️ Dev tip: for auto-reload run: uvicorn main:app --reload --host 127.0.0.1 --port 5000")
        print("ℹ️ UI: http://127.0.0.1:5000/static/index.html")
    uvicorn.run(app, host="127.0.0.1", port=5000, reload=False)
