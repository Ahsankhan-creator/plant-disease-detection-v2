// ============================================
// PlantGuard AI - Smart Farming Assistant
// JavaScript Frontend
// ============================================

// Use relative API base so it works on both localhost and Vercel.
const API_BASE = '/api';

// ============================================
// Utility Functions
// ============================================

function showLoading(text = 'Processing...') {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingOverlay').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.add('hidden');
}

function showElement(id) {
    document.getElementById(id).classList.remove('hidden');
}

function hideElement(id) {
    document.getElementById(id).classList.add('hidden');
}

async function apiRequest(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        
        if (!response.ok) {
            let message = 'API request failed';
            const contentType = response.headers.get('content-type') || '';
            if (contentType.includes('application/json')) {
                const error = await response.json();
                message = error.detail || error.message || message;
            } else {
                const text = await response.text();
                if (text) message = text;
            }
            throw new Error(message);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// ============================================
// Weather Functions
// ============================================

let currentWeatherData = null;

async function getWeather() {
    const location = document.getElementById('locationInput').value.trim();
    if (!location) {
        alert('Please enter a location');
        return;
    }
    
    showLoading('Fetching weather data...');
    
    try {
        // Get current weather
        const weatherData = await apiRequest('/weather', {
            method: 'POST',
            body: JSON.stringify({ location })
        });
        
        currentWeatherData = weatherData;
        displayWeather(weatherData);
        
        // Get forecast
        const forecastData = await apiRequest('/weather/forecast', {
            method: 'POST',
            body: JSON.stringify({ location })
        });
        
        displayForecast(forecastData);
        
    } catch (error) {
        alert('Failed to fetch weather: ' + error.message);
    } finally {
        hideLoading();
    }
}

function displayWeather(data) {
    document.getElementById('locationName').textContent = `${data.location}, ${data.country}`;
    document.getElementById('temperature').textContent = Math.round(data.temperature);
    document.getElementById('weatherDescription').textContent = data.description;
    document.getElementById('weatherIcon').src = `https://openweathermap.org/img/wn/${data.icon}@2x.png`;
    document.getElementById('feelsLike').textContent = `${Math.round(data.feels_like)}°C`;
    document.getElementById('humidity').textContent = `${data.humidity}%`;
    document.getElementById('windSpeed').textContent = `${data.wind_speed} m/s`;
    document.getElementById('visibility').textContent = `${data.visibility} km`;
    
    showElement('weatherResult');
}

function displayForecast(data) {
    const container = document.getElementById('forecastCards');
    container.innerHTML = '';
    
    data.forecasts.forEach(forecast => {
        const date = new Date(forecast.datetime);
        const card = document.createElement('div');
        card.className = 'forecast-card';
        card.innerHTML = `
            <div class="date">${date.toLocaleDateString('en-US', { weekday: 'short', hour: '2-digit' })}</div>
            <img src="https://openweathermap.org/img/wn/${forecast.icon}@2x.png" alt="${forecast.description}">
            <div class="temp">${Math.round(forecast.temperature)}°C</div>
            <div class="rain"><i class="fas fa-tint"></i> ${Math.round(forecast.rain_probability)}%</div>
        `;
        container.appendChild(card);
    });
    
    showElement('forecastResult');
}

// ============================================
// Disease Detection Functions
// ============================================

let selectedFile = null;

function initUploadArea() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('diseaseImage');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length) handleFileSelect(files[0]);
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFileSelect(e.target.files[0]);
    });
    
    document.getElementById('removeImage').addEventListener('click', removeSelectedImage);
}

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }
    
    selectedFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImg').src = e.target.result;
        showElement('imagePreview');
        hideElement('uploadArea');
        document.getElementById('detectDiseaseBtn').disabled = false;
    };
    reader.readAsDataURL(file);
}

function removeSelectedImage() {
    selectedFile = null;
    document.getElementById('previewImg').src = '';
    hideElement('imagePreview');
    showElement('uploadArea');
    document.getElementById('detectDiseaseBtn').disabled = true;
    document.getElementById('diseaseImage').value = '';
    hideElement('diseaseResult');
}

async function detectDisease() {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }
    
    showLoading('Analyzing plant image...');
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        const response = await fetch(`${API_BASE}/detect-disease`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            let message = 'Disease detection failed';
            const contentType = response.headers.get('content-type') || '';
            if (contentType.includes('application/json')) {
                const error = await response.json();
                message = error.detail || error.message || message;
            } else {
                const text = await response.text();
                if (text) message = text;
            }
            throw new Error(message);
        }
        
        const data = await response.json();
        displayDiseaseResult(data);
        
    } catch (error) {
        alert('Disease detection failed: ' + error.message);
    } finally {
        hideLoading();
    }
}

function displayDiseaseResult(data) {
    const statusBadge = document.getElementById('diseaseStatus');
    statusBadge.textContent = data.is_healthy ? 'HEALTHY' : 'DISEASE DETECTED';
    statusBadge.className = `status-badge ${data.is_healthy ? 'healthy' : 'diseased'}`;
    
    document.getElementById('diseaseName').textContent = data.disease;
    document.getElementById('confidenceBar').style.width = `${data.confidence}%`;
    document.getElementById('confidenceText').textContent = `Confidence: ${data.confidence}%`;
    
    // Display top predictions
    const predictionsContainer = document.getElementById('topPredictions');
    predictionsContainer.innerHTML = '<h4>Top Predictions:</h4>';
    
    data.top_predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <span class="name">${index + 1}. ${pred.disease}</span>
            <span class="confidence">${pred.confidence}%</span>
        `;
        predictionsContainer.appendChild(item);
    });
    
    showElement('diseaseResult');

    // AI advice (generated on backend using OpenRouter)
    const panel = document.getElementById('aiAdvicePanel');
    const statusEl = document.getElementById('aiAdviceStatus');
    const textEl = document.getElementById('aiAdviceText');
    const speakBtn = document.getElementById('speakAdviceBtn');

    // Reset
    statusEl.textContent = '';
    textEl.textContent = '';

    if (data.ai_advice_status === 'success' && data.ai_advice) {
        statusEl.textContent = 'Generated by AI based on the model prediction.';
        textEl.textContent = data.ai_advice;
        speakBtn.disabled = false;
        speakBtn.onclick = () => speakText(data.ai_advice);
        showElement('aiAdvicePanel');
    } else if (data.ai_advice_status === 'error') {
        statusEl.textContent = 'AI advice could not be generated.';
        textEl.textContent = data.ai_advice_error || 'Unknown error.';
        speakBtn.disabled = true;
        showElement('aiAdvicePanel');
    } else {
        // If backend is configured to not return advice
        hideElement('aiAdvicePanel');
    }
}

async function speakDiseaseResult() {
    const disease = document.getElementById('diseaseName').textContent;
    const confidence = document.getElementById('confidenceText').textContent;
    const isHealthy = document.getElementById('diseaseStatus').textContent === 'HEALTHY';
    
    let text;
    if (isHealthy) {
        text = `Plant appears healthy. ${confidence}.`;
    } else {
        text = `Detected: ${disease}. ${confidence}.`;
    }
    
    await speakText(text);
}

// ============================================
// Irrigation Functions
// ============================================

function initIrrigationSlider() {
    const slider = document.getElementById('soilMoisture');
    const valueDisplay = document.getElementById('moistureValue');
    
    slider.addEventListener('input', () => {
        valueDisplay.textContent = `${slider.value}%`;
    });
}

async function getIrrigationRecommendation() {
    const cropType = document.getElementById('cropType').value;
    const soilMoisture = parseFloat(document.getElementById('soilMoisture').value);
    const lastWatered = parseInt(document.getElementById('lastWatered').value);
    const useWeatherData = document.getElementById('useWeatherData').checked;
    
    let temperature = 25;
    let humidity = 50;
    
    if (useWeatherData && currentWeatherData) {
        temperature = currentWeatherData.temperature || 25;
        humidity = currentWeatherData.humidity || 50;
    }
    
    showLoading('Calculating irrigation needs...');
    
    try {
        const data = await apiRequest('/irrigation', {
            method: 'POST',
            body: JSON.stringify({
                crop_type: cropType,
                soil_moisture: soilMoisture,
                temperature: temperature,
                humidity: humidity,
                last_watered_hours: lastWatered
            })
        });
        
        displayIrrigationResult(data);
        
    } catch (error) {
        alert('Failed to get irrigation recommendation: ' + error.message);
    } finally {
        hideLoading();
    }
}

function displayIrrigationResult(data) {
    // Update score circle
    const score = data.irrigation_score;
    const circumference = 283;
    const offset = circumference - (score / 100) * circumference;
    
    const scoreCircle = document.getElementById('scoreCircle');
    scoreCircle.style.strokeDashoffset = offset;
    
    // Set color based on action
    const colors = {
        red: '#e53935',
        orange: '#ffa726',
        yellow: '#ffeb3b',
        green: '#43a047'
    };
    scoreCircle.style.stroke = colors[data.action_color] || colors.green;
    
    document.getElementById('irrigationScore').textContent = score;
    document.getElementById('irrigationScore').style.color = colors[data.action_color];
    
    // Update action badge
    const actionBadge = document.getElementById('irrigationAction');
    actionBadge.textContent = data.action;
    actionBadge.className = `irrigation-action ${data.action_color}`;
    
    // Display recommendations
    const recsContainer = document.getElementById('irrigationRecommendations');
    recsContainer.innerHTML = '';
    data.recommendations.forEach(rec => {
        const p = document.createElement('p');
        p.textContent = rec;
        recsContainer.appendChild(p);
    });
    
    // Display crop advice
    const adviceContainer = document.getElementById('cropAdvice');
    adviceContainer.innerHTML = `
        <h4><i class="fas fa-lightbulb"></i> Crop-Specific Advice</h4>
        <p>${data.crop_advice}</p>
    `;
    
    showElement('irrigationResult');
}

// ============================================
// Chat Functions
// ============================================

async function sendMessage(message = null) {
    const input = document.getElementById('chatInput');
    const userMessage = message || input.value.trim();
    
    if (!userMessage) return;
    
    // Clear input
    input.value = '';
    
    // Add user message to chat
    addMessageToChat(userMessage, 'user');
    
    // Show typing indicator
    const typingId = addTypingIndicator();
    
    try {
        const data = await apiRequest('/chat', {
            method: 'POST',
            body: JSON.stringify({ message: userMessage })
        });
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        // Add bot response
        addMessageToChat(data.response, 'bot');
        
    } catch (error) {
        removeTypingIndicator(typingId);
        addMessageToChat('Sorry, I encountered an error. Please try again.', 'bot');
    }
}

function addMessageToChat(content, sender) {
    const chatContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const icon = sender === 'bot' ? 'fa-robot' : 'fa-user';
    
    // Process markdown-like formatting
    let formattedContent = content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>')
        .replace(/• /g, '<br>• ');
    
    messageDiv.innerHTML = `
        <div class="message-avatar"><i class="fas ${icon}"></i></div>
        <div class="message-content">
            <p>${formattedContent}</p>
            ${sender === 'bot' ? `
                <div class="message-actions">
                    <button class="btn-icon speak-btn" title="Read aloud" type="button">
                        <i class="fas fa-volume-up"></i>
                    </button>
                </div>
            ` : ''}
        </div>
    `;

    if (sender === 'bot') {
        const speakButton = messageDiv.querySelector('.speak-btn');
        if (speakButton) {
            speakButton.addEventListener('click', () => speakText(content));
        }
    }
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addTypingIndicator() {
    const chatContainer = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.id = 'typing-' + Date.now();
    typingDiv.innerHTML = `
        <div class="message-avatar"><i class="fas fa-robot"></i></div>
        <div class="message-content">
            <p><i class="fas fa-spinner fa-spin"></i> Thinking...</p>
        </div>
    `;
    chatContainer.appendChild(typingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return typingDiv.id;
}

function removeTypingIndicator(id) {
    const typingDiv = document.getElementById(id);
    if (typingDiv) typingDiv.remove();
}

// ============================================
// Voice Functions
// ============================================

let recognition = null;
let isRecording = false;

function initVoiceInput() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            document.getElementById('chatInput').value = transcript;
            stopRecording();
            sendMessage(transcript);
        };
        
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            stopRecording();
        };
        
        recognition.onend = () => {
            stopRecording();
        };
    }
}

function toggleVoiceInput() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

function startRecording() {
    if (!recognition) {
        alert('Speech recognition is not supported in your browser');
        return;
    }
    
    isRecording = true;
    document.getElementById('voiceInputBtn').classList.add('recording');
    recognition.start();
}

function stopRecording() {
    isRecording = false;
    document.getElementById('voiceInputBtn').classList.remove('recording');
    if (recognition) {
        recognition.stop();
    }
}

async function speakText(text) {
    showLoading('Generating speech...');
    
    try {
        const response = await fetch(`${API_BASE}/tts`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });
        
        if (!response.ok) {
            const errText = await response.text();
            throw new Error(errText || 'TTS request failed');
        }
        
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = document.getElementById('ttsAudio');
        audio.src = audioUrl;

        audio.onended = () => {
            try { URL.revokeObjectURL(audioUrl); } catch (_) {}
        };

        try {
            await audio.play();
        } catch (playErr) {
            // If autoplay/device blocks playback, fallback to browser TTS
            console.warn('Audio playback failed, falling back:', playErr);
            throw playErr;
        }
        
    } catch (error) {
        console.error('TTS Error:', error);
        // Fallback to browser TTS
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.9;
            speechSynthesis.speak(utterance);
        } else {
            alert('Text-to-speech is not available');
        }
    } finally {
        hideLoading();
    }
}

// ============================================
// Navigation Functions
// ============================================

function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });
}

// ============================================
// Event Listeners
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize components
    initUploadArea();
    initIrrigationSlider();
    initVoiceInput();
    initNavigation();
    
    // Weather events
    document.getElementById('getWeatherBtn').addEventListener('click', getWeather);
    document.getElementById('locationInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') getWeather();
    });
    
    // Disease detection events
    document.getElementById('detectDiseaseBtn').addEventListener('click', detectDisease);
    document.getElementById('speakDiseaseBtn').addEventListener('click', speakDiseaseResult);
    
    // Irrigation events
    document.getElementById('getIrrigationBtn').addEventListener('click', getIrrigationRecommendation);
    
    // Chat events
    document.getElementById('sendMessageBtn').addEventListener('click', () => sendMessage());
    document.getElementById('chatInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    document.getElementById('voiceInputBtn').addEventListener('click', toggleVoiceInput);
    
    // Quick message buttons
    document.querySelectorAll('.quick-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const message = btn.dataset.message;
            sendMessage(message);
        });
    });
    
    // Load initial weather
    getWeather();
});
