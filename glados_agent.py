#!/usr/bin/env python3
"""
Agente GLaDOS con voz: microfono (push-to-talk + Whisper) -> Ollama (LLM +
tool calling) -> Piper (voz clonada).

Uso:
    python3 glados_agent.py
    Presiona Enter para empezar a grabar, habla, presiona Enter de nuevo
    para terminar la grabacion.

Requiere (en el venv de piper-train):
    pip install sounddevice faster-whisper numpy ddgs
    (y el paquete de sistema 'portaudio')
"""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS  # nombre viejo del paquete

# --- Configuracion -------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "glados-ministral:latest"

PIPER_BIN = "/mnt/games/AI-Assistant/piper-train/venv/bin/piper"
VOICE_MODEL = "/mnt/games/AI-Assistant/models-final/es_ES-glados-medium.onnx"
PLAYBACK_CMD = ["paplay"]

WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

SAMPLE_RATE = 16000

# --- Definicion de herramientas ------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Obtiene el clima actual para una ciudad o ubicacion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Nombre de la ciudad, ej: 'Ciudad de Mexico'",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Busca informacion actual en internet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Terminos de busqueda",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        # NUEVO
        "type": "function",
        "function": {
            "name": "check_disk_usage",
            "description": "Muestra el espacio usado y disponible en todos los discos y particiones montados en el sistema.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        # NUEVO
        "type": "function",
        "function": {
            "name": "check_memory_usage",
            "description": "Muestra la memoria RAM usada y disponible del sistema.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

REQUIRES_CONFIRMATION: set[str] = set()


# --- Implementacion de herramientas ---------------------------------------

def get_weather(location: str) -> str:
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "es"},
        timeout=10,
    ).json()

    results = geo.get("results")
    if not results:
        return f"No encontre la ubicacion '{location}'."

    place = results[0]
    lat, lon = place["latitude"], place["longitude"]
    nombre = place.get("name", location)

    forecast = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "timezone": "auto",
        },
        timeout=10,
    ).json()

    current = forecast.get("current", {})
    temp = current.get("temperature_2m")
    viento = current.get("wind_speed_10m")

    return f"En {nombre}: temperatura actual {temp} grados C, viento {viento} km/h."


def web_search(query: str) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))

    if not results:
        return "No encontre resultados."

    lines = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        lines.append(f"- {title}: {body}")

    return "\n".join(lines)


AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "web_search": web_search,
}

# --- Voz: grabacion + transcripcion ---------------------------------------

_whisper_model: WhisperModel | None = None


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        print("Cargando modelo de reconocimiento de voz...")
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
        )
    return _whisper_model


def record_audio() -> np.ndarray:
    frames: list[np.ndarray] = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
    )
    stream.start()
    input()
    stream.stop()
    stream.close()

    if not frames:
        return np.array([], dtype=np.float32)
    return np.concatenate(frames, axis=0).flatten()


def transcribe(audio: np.ndarray) -> str:
    model = get_whisper_model()
    segments, _ = model.transcribe(audio, language="es")
    return " ".join(seg.text.strip() for seg in segments).strip()


# -------------------------------------------------------------------------

conversation_history: list[dict] = []


def call_ollama(messages: list[dict]) -> dict:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]


def confirm_action(tool_name: str, args: dict) -> bool:
    print(f"\n[GLaDOS quiere ejecutar: {tool_name}({args})]")
    resp = input("Confirmar? (s/n): ").strip().lower()
    return resp in ("s", "si", "sí", "y", "yes")


def ask_glados(user_text: str) -> str:
    conversation_history.append({"role": "user", "content": user_text})

    while True:
        message = call_ollama(conversation_history)
        conversation_history.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return message.get("content", "")

        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = call["function"].get("arguments", {})
            if isinstance(fn_args, str):
                fn_args = json.loads(fn_args)

            if fn_name in REQUIRES_CONFIRMATION and not confirm_action(fn_name, fn_args):
                result = "El usuario cancelo esta accion."
            else:
                fn = AVAILABLE_FUNCTIONS.get(fn_name)
                if fn is None:
                    result = f"Herramienta desconocida: {fn_name}"
                else:
                    try:
                        result = fn(**fn_args)
                    except Exception as exc:  # noqa: BLE001
                        result = f"Error ejecutando {fn_name}: {exc}"

            conversation_history.append(
                {"role": "tool", "name": fn_name, "content": result}
            )


def speak(text: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        subprocess.run(
            [PIPER_BIN, "-m", VOICE_MODEL, "--output_file", wav_path],
            input=text.encode("utf-8"),
            check=True,
        )
        subprocess.run(PLAYBACK_CMD + [wav_path], check=True)
    finally:
        Path(wav_path).unlink(missing_ok=True)


def main() -> None:
    get_whisper_model()
    print("\nGLaDOS esta en linea.")
    print("Presiona Enter para hablar (Enter de nuevo para terminar), o escribi 'salir'.\n")

    while True:
        try:
            trigger = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFinalizando protocolo de prueba.")
            break

        if trigger.lower() in ("salir", "exit", "quit"):
            print("Finalizando protocolo de prueba.")
            break

        print("Escuchando... (Enter para cortar)")
        audio = record_audio()
        if audio.size == 0:
            print("No se capturo audio.")
            continue

        print("Transcribiendo...")
        user_text = transcribe(audio)
        if not user_text:
            print("No entendi nada, proba de nuevo.")
            continue

        print(f"Vos: {user_text}")

        try:
            reply = ask_glados(user_text)
        except requests.exceptions.ConnectionError:
            print("No pude conectarme a Ollama. Esta corriendo el servicio?")
            continue
        except requests.exceptions.RequestException as exc:
            print(f"Error al consultar Ollama: {exc}")
            continue

        print(f"GLaDOS: {reply}\n")

        try:
            speak(reply)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"(No se pudo reproducir el audio: {exc})")


if __name__ == "__main__":
    main()
