# 🌱 PlantGuard AI - Smart Farming Assistant

An AI-powered smart farming application featuring weather prediction, plant disease detection, smart irrigation recommendations, and a voice-enabled crop doctor chatbot.

![PlantGuard AI](https://img.shields.io/badge/PlantGuard-AI-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-teal?style=for-the-badge)

## ✨ Features

### 🌤️ Weather Prediction
- Real-time weather data for any location
- 5-day weather forecast
- Temperature, humidity, wind speed, and visibility
- Rainfall probability predictions

### 🔬 Plant Disease Detection
- AI-powered disease detection using ResNet50 model
- Upload plant leaf images for instant analysis
- Top 3 disease predictions with confidence scores
- Voice readout of results

### 💧 Smart Irrigation Advisor
- Intelligent irrigation recommendations
- Considers soil moisture, temperature, humidity
- Crop-specific watering advice
- Integration with real-time weather data

### 🤖 AI Crop Doctor
- ChatGPT-like conversational interface
- Expert advice on plant diseases, pests, and crops
- Voice input using speech recognition
- Text-to-speech output using ElevenLabs

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "AI PROJECT V2"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

5. **Open your browser:**
   Navigate to `http://localhost:5000/static/index.html`

## 🔒 Keep API keys private (GitHub)

Do **not** put API keys inside `config.py` as plain text. This project reads keys from environment variables.

### Environment variables you should set

- `OPENROUTER_API_KEY`
- `WEATHER_API_KEY`
- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID` (optional)
- `HF_MODEL_ID` (optional, default is `mesabo/agri-plant-disease-resnet50`)

Disease detection runs locally by default and requires `torch` + `transformers`.

If you ever committed keys in the past, rotate them and remove them from git history (or create a fresh repo) before making the repo public.

## 📁 Project Structure

```
AI PROJECT V2/
├── config.py           # API keys and configuration
├── main.py             # FastAPI backend server
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── static/
    ├── index.html      # Main HTML page
    ├── styles.css      # CSS styling
    └── script.js       # JavaScript frontend
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and app info |
| `/api/weather` | POST | Get current weather |
| `/api/weather/forecast` | POST | Get 5-day forecast |
| `/api/detect-disease` | POST | Detect plant disease from image |
| `/api/chat` | POST | Chat with AI Crop Doctor |
| `/api/tts` | POST | Text-to-speech conversion |
| `/api/irrigation` | POST | Get irrigation recommendations |
| `/api/full-analysis` | POST | Complete farm analysis |

## 🔑 API Services Used

- **OpenRouter API** - AI chat (DeepSeek R1T Chimera model)
- **OpenWeatherMap** - Weather data
- **ElevenLabs** - Text-to-speech
- **Hugging Face** - Plant disease detection model

## 📱 Features Overview

### Weather Section
- Enter any city name to get weather data
- View current conditions and 5-day forecast
- Integrated with irrigation recommendations

### Disease Detection
- Drag & drop or browse to upload plant images
- Get instant AI-powered disease diagnosis
- View confidence scores and alternatives
- Voice readout of results

### Irrigation Advisor
- Select your crop type
- Adjust soil moisture levels
- Set hours since last watering
- Get personalized recommendations

### AI Crop Doctor
- Ask questions about farming, pests, diseases
- Use voice input for hands-free operation
- Quick-access preset questions
- Text-to-speech responses

## 🛠️ Technology Stack

- **Backend:** FastAPI, Python
- **Frontend:** HTML5, CSS3, JavaScript
- **AI/ML:** PyTorch, Transformers, ResNet50
- **APIs:** OpenRouter, OpenWeatherMap, ElevenLabs
- **Styling:** Custom CSS with Font Awesome icons

## 📝 Notes

- The plant disease detection model downloads on first run (~100MB)
- Voice input requires a modern browser with Web Speech API support
- Text-to-speech uses ElevenLabs API with browser fallback

## 👨‍💻 Author

Created for smart farming and agricultural assistance.

## 📄 License

This project is for educational and research purposes.

---

**🌾 Happy Farming with AI! 🌾**
