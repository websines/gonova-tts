# Integration Guide

How to integrate the ASR and TTS microservices into your applications.

## Table of Contents

1. [Simple Voice Bot (No LLM)](#simple-voice-bot-no-llm)
2. [Voice Agent with LLM](#voice-agent-with-llm)
3. [Website Integration](#website-integration)
4. [Phone System (Twilio)](#phone-system-twilio)
5. [Mobile App Integration](#mobile-app-integration)
6. [Custom Integrations](#custom-integrations)

---

## Simple Voice Bot (No LLM)

Direct audio-to-audio conversion without any intelligence in between.

### Architecture

```
Microphone → ASR Service → TTS Service → Speaker
```

### Python Example

```python
import asyncio
import websockets
import pyaudio

# Audio config
SAMPLE_RATE = 16000
CHUNK_SIZE = 4096

async def simple_voice_bot():
    # Connect to both services
    asr_ws = await websockets.connect('ws://localhost:8001/v1/stream/asr')
    tts_ws = await websockets.connect('ws://localhost:8002/v1/stream/tts')

    # Setup audio I/O
    audio = pyaudio.PyAudio()

    # Input stream (microphone)
    input_stream = audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    # Output stream (speaker)
    output_stream = audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=24000,
        output=True
    )

    async def capture_and_transcribe():
        """Capture audio and send to ASR"""
        while True:
            # Read from microphone
            audio_chunk = input_stream.read(CHUNK_SIZE)
            await asr_ws.send(audio_chunk)

    async def receive_and_speak():
        """Receive transcriptions and synthesize"""
        async for message in asr_ws:
            data = json.loads(message)

            if data['type'] == 'final':
                # Get transcription
                text = data['text']
                print(f"Heard: {text}")

                # Send to TTS
                await tts_ws.send(json.dumps({
                    'type': 'synthesize',
                    'text': text,
                    'streaming': True
                }))

    async def play_audio():
        """Play TTS audio"""
        async for message in tts_ws:
            if isinstance(message, bytes):
                output_stream.write(message)

    # Run all tasks
    await asyncio.gather(
        capture_and_transcribe(),
        receive_and_speak(),
        play_audio()
    )

# Run
asyncio.run(simple_voice_bot())
```

### Use Cases
- Echo/Parrot bot
- Voice testing
- Audio format conversion
- Accent/voice transformation

---

## Voice Agent with LLM

Add intelligence between ASR and TTS using any LLM.

### Architecture

```
Microphone → ASR → LLM (OpenAI/Claude/Local) → TTS → Speaker
```

### Python Example (with OpenAI)

```python
import asyncio
import websockets
from openai import AsyncOpenAI

openai_client = AsyncOpenAI(api_key="your-api-key")

async def voice_agent_with_llm():
    # Connect to services
    asr_ws = await websockets.connect('ws://localhost:8001/v1/stream/asr')
    tts_ws = await websockets.connect('ws://localhost:8002/v1/stream/tts')

    # Conversation history
    messages = [
        {"role": "system", "content": "You are a helpful voice assistant."}
    ]

    # Capture audio and transcribe
    async def process_audio():
        async for message in asr_ws:
            data = json.loads(message)

            if data['type'] == 'final':
                user_text = data['text']
                print(f"User: {user_text}")

                # Add to history
                messages.append({"role": "user", "content": user_text})

                # Get LLM response
                response = await openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=messages
                )

                assistant_text = response.choices[0].message.content
                print(f"Assistant: {assistant_text}")

                # Add to history
                messages.append({"role": "assistant", "content": assistant_text})

                # Synthesize response
                await tts_ws.send(json.dumps({
                    'type': 'synthesize',
                    'text': assistant_text,
                    'streaming': True
                }))

    # Play TTS audio
    async def play_audio():
        output_stream = setup_audio_output()

        async for message in tts_ws:
            if isinstance(message, bytes):
                output_stream.write(message)

    # Run
    await asyncio.gather(
        capture_microphone_and_send(asr_ws),
        process_audio(),
        play_audio()
    )

asyncio.run(voice_agent_with_llm())
```

### With Memory (Redis)

```python
import redis.asyncio as redis

# Connect to Redis
redis_client = redis.Redis(host='localhost', port=6379)

async def get_conversation_history(user_id: str):
    """Load conversation from Redis"""
    history = await redis_client.lrange(f"conv:{user_id}", 0, -1)
    return [json.loads(msg) for msg in history]

async def save_message(user_id: str, role: str, content: str):
    """Save message to Redis"""
    message = json.dumps({"role": role, "content": content})
    await redis_client.rpush(f"conv:{user_id}", message)

    # Keep only last 20 messages
    await redis_client.ltrim(f"conv:{user_id}", -20, -1)

# Usage in voice agent
async def voice_agent_with_memory(user_id: str):
    # Load history
    messages = await get_conversation_history(user_id)

    # ... process audio ...

    # When user speaks
    await save_message(user_id, "user", user_text)

    # Get LLM response
    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    # Save assistant response
    await save_message(user_id, "assistant", assistant_text)
```

### With Anthropic Claude

```python
from anthropic import AsyncAnthropic

anthropic_client = AsyncAnthropic(api_key="your-api-key")

async def voice_agent_claude():
    asr_ws = await websockets.connect('ws://localhost:8001/v1/stream/asr')
    tts_ws = await websockets.connect('ws://localhost:8002/v1/stream/tts')

    async for message in asr_ws:
        data = json.loads(message)

        if data['type'] == 'final':
            # Get Claude response
            response = await anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": data['text']}
                ]
            )

            assistant_text = response.content[0].text

            # Send to TTS
            await tts_ws.send(json.dumps({
                'type': 'synthesize',
                'text': assistant_text,
                'streaming': True
            }))
```

### With Local LLM (Ollama)

```python
import httpx

async def call_local_llm(text: str):
    """Call local Ollama instance"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3',
                'prompt': text,
                'stream': False
            }
        )
        return response.json()['response']

# Usage
async def voice_agent_local():
    asr_ws = await websockets.connect('ws://localhost:8001/v1/stream/asr')
    tts_ws = await websockets.connect('ws://localhost:8002/v1/stream/tts')

    async for message in asr_ws:
        data = json.loads(message)

        if data['type'] == 'final':
            # Call local LLM
            response = await call_local_llm(data['text'])

            # Send to TTS
            await tts_ws.send(json.dumps({
                'type': 'synthesize',
                'text': response,
                'streaming': True
            }))
```

---

## Website Integration

Add voice capabilities to your website.

### Architecture

```
Browser (Mic) → WebSocket → ASR → Your Backend → TTS → Browser (Speaker)
```

### Frontend (HTML + JavaScript)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Voice Chat</title>
</head>
<body>
    <button id="startBtn">Start Voice Chat</button>
    <button id="stopBtn" disabled>Stop</button>
    <div id="transcript"></div>
    <div id="response"></div>

    <script>
        let asrSocket = null;
        let ttsSocket = null;
        let audioContext = null;
        let mediaStream = null;
        let processor = null;

        document.getElementById('startBtn').onclick = async () => {
            // Connect to ASR
            asrSocket = new WebSocket('ws://localhost:8001/v1/stream/asr');

            // Connect to TTS
            ttsSocket = new WebSocket('ws://localhost:8002/v1/stream/tts');

            // Setup audio context
            audioContext = new AudioContext({ sampleRate: 16000 });

            // Get microphone access
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const source = audioContext.createMediaStreamSource(mediaStream);

            // Create audio processor
            processor = audioContext.createScriptProcessor(4096, 1, 1);

            processor.onaudioprocess = (e) => {
                const audioData = e.inputBuffer.getChannelData(0);

                // Send to ASR
                if (asrSocket.readyState === WebSocket.OPEN) {
                    asrSocket.send(audioData);
                }
            };

            source.connect(processor);
            processor.connect(audioContext.destination);

            // Receive transcriptions
            asrSocket.onmessage = async (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'final') {
                    document.getElementById('transcript').innerText =
                        `You: ${data.text}`;

                    // Send to your backend for LLM processing
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: data.text })
                    });

                    const result = await response.json();

                    document.getElementById('response').innerText =
                        `Bot: ${result.message}`;

                    // Send to TTS
                    ttsSocket.send(JSON.stringify({
                        type: 'synthesize',
                        text: result.message,
                        streaming: true
                    }));
                }
            };

            // Play TTS audio
            const outputContext = new AudioContext({ sampleRate: 24000 });

            ttsSocket.onmessage = async (event) => {
                if (event.data instanceof Blob) {
                    const arrayBuffer = await event.data.arrayBuffer();
                    const audioBuffer = await outputContext.decodeAudioData(arrayBuffer);

                    const source = outputContext.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(outputContext.destination);
                    source.start();
                }
            };

            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
        };

        document.getElementById('stopBtn').onclick = () => {
            // Cleanup
            if (asrSocket) asrSocket.close();
            if (ttsSocket) ttsSocket.close();
            if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
            if (processor) processor.disconnect();
            if (audioContext) audioContext.close();

            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        };
    </script>
</body>
</html>
```

### Backend (FastAPI)

```python
from fastapi import FastAPI, WebSocket
from openai import AsyncOpenAI

app = FastAPI()
openai_client = AsyncOpenAI()

# In-memory session storage (use Redis in production)
sessions = {}

@app.post("/api/chat")
async def chat(message: dict):
    """Process chat messages with LLM"""
    user_message = message['message']

    # Get LLM response
    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
    )

    return {"message": response.choices[0].message.content}

# Or use WebSocket for full control
@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    await websocket.accept()

    # Connect to ASR and TTS
    asr_ws = await websockets.connect('ws://localhost:8001/v1/stream/asr')
    tts_ws = await websockets.connect('ws://localhost:8002/v1/stream/tts')

    async def forward_audio():
        """Forward audio from browser to ASR"""
        async for message in websocket.iter_bytes():
            await asr_ws.send(message)

    async def process_transcriptions():
        """Process ASR → LLM → TTS"""
        async for message in asr_ws:
            data = json.loads(message)

            if data['type'] == 'final':
                # LLM call
                response = await openai_client.chat.completions.create(...)

                # Send to TTS
                await tts_ws.send(json.dumps({
                    'type': 'synthesize',
                    'text': response.choices[0].message.content
                }))

    async def forward_audio_back():
        """Forward TTS audio to browser"""
        async for message in tts_ws:
            if isinstance(message, bytes):
                await websocket.send_bytes(message)

    await asyncio.gather(
        forward_audio(),
        process_transcriptions(),
        forward_audio_back()
    )
```

---

## Phone System (Twilio)

Integrate with Twilio for phone calls.

### Architecture

```
Phone Call → Twilio → Your Server → ASR/TTS → Response → Twilio → Phone
```

### Twilio WebSocket Integration

```python
from fastapi import FastAPI, WebSocket, Request
from twilio.twiml.voice_response import VoiceResponse, Connect

app = FastAPI()

@app.post("/voice/incoming")
async def incoming_call(request: Request):
    """Handle incoming Twilio call"""
    response = VoiceResponse()

    # Connect call to WebSocket
    connect = Connect()
    connect.stream(url=f'wss://{request.url.hostname}/voice/stream')
    response.append(connect)

    return str(response)

@app.websocket("/voice/stream")
async def voice_stream(websocket: WebSocket):
    """Handle Twilio media stream"""
    await websocket.accept()

    # Connect to ASR and TTS
    asr_ws = await websockets.connect('ws://localhost:8001/v1/stream/asr')
    tts_ws = await websockets.connect('ws://localhost:8002/v1/stream/tts')

    async for message in websocket.iter_text():
        data = json.loads(message)

        if data['event'] == 'media':
            # Decode Twilio audio (mulaw, 8kHz)
            audio_payload = data['media']['payload']
            audio_bytes = base64.b64decode(audio_payload)

            # Convert mulaw to PCM
            pcm_audio = audioop.ulaw2lin(audio_bytes, 2)

            # Resample 8kHz → 16kHz
            resampled = resample_audio(pcm_audio, 8000, 16000)

            # Send to ASR
            await asr_ws.send(resampled)

        elif data['event'] == 'start':
            # Call started
            stream_sid = data['start']['streamSid']

    # Process ASR transcriptions
    async def process_call():
        async for asr_message in asr_ws:
            asr_data = json.loads(asr_message)

            if asr_data['type'] == 'final':
                # Get LLM response
                llm_response = await call_your_llm(asr_data['text'])

                # Send to TTS
                await tts_ws.send(json.dumps({
                    'type': 'synthesize',
                    'text': llm_response,
                    'streaming': True
                }))

    # Send TTS audio back to Twilio
    async def stream_audio():
        async for tts_message in tts_ws:
            if isinstance(tts_message, bytes):
                # Resample 24kHz → 8kHz
                resampled = resample_audio(tts_message, 24000, 8000)

                # Convert to mulaw
                mulaw_audio = audioop.lin2ulaw(resampled, 2)

                # Encode for Twilio
                encoded = base64.b64encode(mulaw_audio).decode()

                # Send via WebSocket
                await websocket.send_json({
                    'event': 'media',
                    'streamSid': stream_sid,
                    'media': {
                        'payload': encoded
                    }
                })

    await asyncio.gather(process_call(), stream_audio())
```

### Twilio Configuration

```python
# Twilio webhook URL
# https://your-domain.com/voice/incoming

# Required Twilio packages
# pip install twilio
```

---

## Mobile App Integration

### React Native Example

```javascript
import { WebSocket } from 'react-native';
import AudioRecorderPlayer from 'react-native-audio-recorder-player';

const VoiceAgent = () => {
  const [asrSocket, setAsrSocket] = useState(null);
  const [ttsSocket, setTtsSocket] = useState(null);
  const audioRecorderPlayer = new AudioRecorderPlayer();

  const startVoiceChat = async () => {
    // Connect to ASR
    const asr = new WebSocket('ws://your-server:8001/v1/stream/asr');
    setAsrSocket(asr);

    // Connect to TTS
    const tts = new WebSocket('ws://your-server:8002/v1/stream/tts');
    setTtsSocket(tts);

    // Start recording
    await audioRecorderPlayer.startRecorder();

    // Send audio chunks
    audioRecorderPlayer.addRecordBackListener((e) => {
      const audioData = e.currentMetering;  // Get audio data
      if (asr.readyState === WebSocket.OPEN) {
        asr.send(audioData);
      }
    });

    // Receive transcriptions
    asr.onmessage = async (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'final') {
        // Call your LLM backend
        const response = await fetch('https://your-api.com/chat', {
          method: 'POST',
          body: JSON.stringify({ message: data.text })
        });

        const result = await response.json();

        // Send to TTS
        tts.send(JSON.stringify({
          type: 'synthesize',
          text: result.message,
          streaming: true
        }));
      }
    };

    // Play TTS audio
    tts.onmessage = (event) => {
      if (typeof event.data !== 'string') {
        // Audio chunk - play it
        audioRecorderPlayer.startPlayer(event.data);
      }
    };
  };

  return (
    <View>
      <Button title="Start Voice Chat" onPress={startVoiceChat} />
    </View>
  );
};
```

---

## Custom Integrations

### Discord Bot

```python
import discord
from discord import FFmpegPCMAudio

intents = discord.Intents.default()
intents.voice_states = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Bot ready: {client.user}')

@client.event
async def on_voice_state_update(member, before, after):
    if after.channel:
        # User joined voice channel
        voice_client = await after.channel.connect()

        # Connect to ASR/TTS
        asr_ws = await websockets.connect('ws://localhost:8001/v1/stream/asr')
        tts_ws = await websockets.connect('ws://localhost:8002/v1/stream/tts')

        # Capture voice
        voice_client.listen(
            discord.AudioSink(callback=lambda data: asr_ws.send(data))
        )

        # Process and respond
        async for message in asr_ws:
            data = json.loads(message)

            if data['type'] == 'final':
                # Get LLM response
                response = await your_llm(data['text'])

                # Synthesize
                await tts_ws.send(json.dumps({
                    'type': 'synthesize',
                    'text': response
                }))

        # Play TTS
        async for audio in tts_ws:
            if isinstance(audio, bytes):
                voice_client.play(FFmpegPCMAudio(audio))

client.run('YOUR_BOT_TOKEN')
```

### Slack Bot

```python
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(token="your-slack-bot-token")

@app.event("app_mention")
async def handle_mention(event, say):
    """Respond to @mentions with voice"""
    text = event['text']

    # Remove mention
    clean_text = text.split('>', 1)[1].strip()

    # Connect to ASR (if processing voice messages)
    # Or just use TTS for text responses

    # Get LLM response
    response = await your_llm(clean_text)

    # Synthesize
    tts_ws = await websockets.connect('ws://localhost:8002/v1/stream/tts')
    await tts_ws.send(json.dumps({
        'type': 'synthesize',
        'text': response
    }))

    # Upload audio to Slack
    audio_chunks = []
    async for chunk in tts_ws:
        if isinstance(chunk, bytes):
            audio_chunks.append(chunk)

    audio_file = b''.join(audio_chunks)

    # Upload
    app.client.files_upload(
        channels=event['channel'],
        file=audio_file,
        filename="response.wav",
        title="Voice Response"
    )

SocketModeHandler(app, "your-app-token").start()
```

### Telegram Bot

```python
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

async def handle_voice(update: Update, context):
    """Handle voice messages"""
    # Download voice message
    voice_file = await update.message.voice.get_file()
    audio_bytes = await voice_file.download_as_bytearray()

    # Send to ASR
    asr_ws = await websockets.connect('ws://localhost:8001/v1/stream/asr')
    await asr_ws.send(audio_bytes)

    # Get transcription
    async for message in asr_ws:
        data = json.loads(message)

        if data['type'] == 'final':
            transcription = data['text']

            # Get LLM response
            response = await your_llm(transcription)

            # Send to TTS
            tts_ws = await websockets.connect('ws://localhost:8002/v1/stream/tts')
            await tts_ws.send(json.dumps({
                'type': 'synthesize',
                'text': response
            }))

            # Collect audio
            audio_chunks = []
            async for chunk in tts_ws:
                if isinstance(chunk, bytes):
                    audio_chunks.append(chunk)

            # Send voice response
            audio_file = b''.join(audio_chunks)
            await update.message.reply_voice(voice=audio_file)
            break

# Setup bot
app = Application.builder().token("YOUR_BOT_TOKEN").build()
app.add_handler(MessageHandler(filters.VOICE, handle_voice))
app.run_polling()
```

---

## Best Practices

### Connection Management
```python
# Always handle reconnections
async def connect_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            ws = await websockets.connect(url)
            return ws
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
```

### Error Handling
```python
# Handle service failures gracefully
try:
    await asr_ws.send(audio_data)
except websockets.exceptions.ConnectionClosed:
    # Reconnect
    asr_ws = await connect_with_retry('ws://localhost:8001/v1/stream/asr')
```

### Performance
```python
# Use connection pooling
from aiohttp import ClientSession

session = ClientSession()

# Reuse WebSocket connections
# Don't reconnect for every request
```

### Security
```python
# Use WSS in production
asr_ws = await websockets.connect(
    'wss://your-domain.com/v1/stream/asr',
    extra_headers={'Authorization': f'Bearer {api_key}'}
)
```

---

## Deployment Considerations

### Load Balancing
```nginx
# nginx.conf
upstream asr_backend {
    server localhost:8001;
    server localhost:8011;  # Second instance
    server localhost:8021;  # Third instance
}

upstream tts_backend {
    server localhost:8002;
    server localhost:8012;
    server localhost:8022;
}

server {
    listen 80;

    location /v1/stream/asr {
        proxy_pass http://asr_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /v1/stream/tts {
        proxy_pass http://tts_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Monitoring
```python
# Add metrics
from prometheus_client import Counter, Histogram

transcription_counter = Counter('asr_transcriptions_total', 'Total transcriptions')
synthesis_counter = Counter('tts_syntheses_total', 'Total syntheses')
latency_histogram = Histogram('asr_latency_seconds', 'ASR latency')

# Instrument your code
with latency_histogram.time():
    result = await transcribe(audio)
transcription_counter.inc()
```

---

## License

See main project LICENSE file (MIT).
