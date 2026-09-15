#!/usr/bin/env python3
"""
Asistente GLaDOS: Ollama (LLM con personalidad) + Piper (voz clonada afinada).

Uso:
    python3 glados_assistant.py

Requiere:
    - Ollama corriendo localmente (systemd service) con el modelo 'glados:latest'
    - El binario de Piper del venv de piper-train (ya compilado/instalado)
    - El modelo ONNX exportado en models-final/
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

# --- Configuración -----------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "glados:latest"

PIPER_BIN = "/mnt/games/AI-Assistant/piper-train/venv/bin/piper"
VOICE_MODEL = "/mnt/games/AI-Assistant/models-final/es_ES-glados-medium.onnx"

# Comando para reproducir audio. paplay (PipeWire/PulseAudio) es lo más
# común en KDE Plasma. Si no funciona, alternativas: ["ffplay", "-nodisp",
# "-autoexit"] o ["aplay"].
PLAYBACK_CMD = ["paplay"]

# -------------------------------------------------------------------------

conversation_history: list[dict[str, str]] = []


def ask_glados(user_text: str) -> str:
    """Envía el mensaje a Ollama y devuelve la respuesta de texto de GLaDOS."""
    conversation_history.append({"role": "user", "content": user_text})

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": conversation_history,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    reply = data["message"]["content"]

    conversation_history.append({"role": "assistant", "content": reply})
    return reply


def speak(text: str) -> None:
    """Sintetiza el texto con Piper y lo reproduce."""
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
    print("GLaDOS está en línea. Escribí 'salir' para terminar la prueba.\n")

    while True:
        try:
            user_text = input("Vos: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFinalizando protocolo de prueba.")
            break

        if not user_text:
            continue
        if user_text.lower() in ("salir", "exit", "quit"):
            print("Finalizando protocolo de prueba.")
            break

        try:
            reply = ask_glados(user_text)
        except requests.exceptions.ConnectionError:
            print("No pude conectarme a Ollama. ¿Está corriendo el servicio?")
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
