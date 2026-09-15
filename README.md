# Asistente GLaDOS — LLM local + voz clonada + agente con herramientas

Asistente conversacional local inspirado en GLaDOS (Portal / Portal 2), con
personalidad propia corriendo en Ollama, voz clonada mediante fine-tuning
directo de un modelo TTS (no conversión de timbre), reconocimiento de voz
por micrófono, y capacidad de usar herramientas (clima, búsqueda web,
estado del sistema).

Todo corre 100% local, sin servicios en la nube, en una RTX 4060 (8GB VRAM).

---

## Índice

1. [Arquitectura general](#arquitectura-general)
2. [Infraestructura base](#infraestructura-base)
3. [Personalidad (Ollama)](#personalidad-ollama)
4. [Pipeline de voz](#pipeline-de-voz)
   - [Etapa 1: RVC (conversión de timbre) — descartada como método principal](#etapa-1-rvc)
   - [Etapa 2: Fine-tuning directo de Piper (método final)](#etapa-2-fine-tuning-piper)
5. [Problemas encontrados y soluciones](#problemas-encontrados-y-soluciones)
6. [El agente: herramientas y reconocimiento de voz](#el-agente)
7. [Estructura de carpetas](#estructura-de-carpetas)
8. [Cómo corrrerlo](#como-correrlo)
9. [Demo de audio](#demo-de-audio)
10. [Nota legal / assets con copyright](#nota-legal)
11. [Créditos](#creditos)

---

## Arquitectura general

```
Microfono
   │
   ▼
Whisper (faster-whisper, push-to-talk) ── transcribe a texto
   │
   ▼
Ollama (glados-ministral, basado en ministral-3:8b)
   │  - Personalidad GLaDOS via Modelfile
   │  - Tool calling nativo: clima, busqueda web, disco, memoria
   ▼
Texto de respuesta (limpiado de markdown/asteriscos)
   │
   ▼
Piper (modelo TTS afinado: es_ES-glados-medium.onnx)
   │
   ▼
Parlantes (paplay)
```

Hardware usado: RTX 4060 (8GB VRAM), CachyOS (Arch Linux), shell Fish.

---

## Infraestructura base

Todo el proyecto vive en un segundo disco (`/mnt/games/AI-Assistant/`) para
no ocupar espacio de la distro principal.

**Fix de permisos necesario:**
- El disco es NTFS montado con `ntfs-3g`. Se cambió `dmask=022` →
  `dmask=000` en `/etc/fstab` para que servicios corriendo bajo otros
  usuarios pudieran escribir carpetas nuevas ahí.
- El servicio `ollama` corría como usuario de sistema `ollama` en vez del
  usuario normal, causando errores de permisos al crear modelos. Se
  resolvió con `systemctl edit ollama`:
  ```ini
  [Service]
  User=<tu_usuario>
  Group=<tu_usuario>
  SupplementaryGroups=video render
  ```

---

## Personalidad (Ollama)

La personalidad vive en un `Modelfile` (`modelfiles/Modelfile.glados-ministral`),
que define:
- **Base:** `ministral-3:8b` (se probó primero con `qwen2.5:7b-instruct-q4_K_M`,
  reemplazado luego por Ministral por mejor calidad de conversación).
- **`SYSTEM`:** define tono (frialdad quirúrgica, sarcasmo pasivo-agresivo,
  obsesión científica, humor negro), con ejemplos de estilo (no diálogo
  copiado del juego) y una instrucción explícita para no usar markdown ni
  asteriscos de acción (`*sonríe*`), ya que el texto se lee literal por el
  sintetizador de voz.
- **`PARAMETER`:** `temperature 0.85`, `top_p 0.9`, `repeat_penalty 1.1`.

```bash
ollama create glados-ministral -f modelfiles/Modelfile.glados-ministral
ollama run glados-ministral:latest
```

### Elegir el modelo base correcto

Con 8GB de VRAM, no todos los modelos descargados sirven. Dos criterios:

1. **Tamaño:** el modelo debe entrar completo en VRAM (dejando margen para
   el contexto de conversación). Como referencia con esta GPU: modelos de
   hasta ~6-7GB en disco entran cómodos; 8GB+ empiezan a desbordar a CPU.
2. **Soporte de tool-calling:** no todos los modelos manejan bien el
   formato de herramientas de Ollama. Confirmado que **no** lo soportan
   bien (o directamente lo rechazan): `deepseek-r1` (las variantes de
   razonamiento de la librería estándar de Ollama devuelven error
   `"does not support tools"`), y modelos antiguos anteriores a la
   convención de tools en Ollama (ej. `deepseek-coder` v1). `qwen2.5`,
   `qwen3.x`, `ministral-3` e `ibm/granite4` sí lo soportan bien.

---

## Pipeline de voz

### Etapa 1: RVC (conversión de timbre) — descartada como método principal {#etapa-1-rvc}

Primer intento: convertir el timbre de una voz TTS genérica (Piper base)
a la de GLaDOS usando RVC (repo Applio, fork de RVC).

- Dataset: audio en español extraído de la copia legítima de Portal 2
  (`sound/vo/glados/` dentro del VPK), 1112 archivos / 94 minutos.
- Entrenamiento: 200 épocas, ~9h24min en RTX 4060.
- **Problema de fondo:** RVC solo cambia el *timbre* de un audio ya
  generado. La cadencia, pausas y modulación real de la actriz nunca
  estuvieron en el audio base genérico de Piper, así que RVC no puede
  "inventar" una prosodia que nunca existió en el audio original. El
  resultado nunca terminaba de sonar 100% natural por más que se
  ajustara el efecto de audio (EQ, saturación, bitcrush) encima.
- **Conclusión:** se mantiene como artefacto opcional/experimental, pero
  el pipeline final **no lo usa**. Ver nota en Etapa 2.

### Etapa 2: Fine-tuning directo de Piper (método final) {#etapa-2-fine-tuning-piper}

En vez de convertir timbre después, se afinó directamente un modelo TTS
(arquitectura VITS, vía [`piper1-gpl`](https://github.com/OHF-Voice/piper1-gpl))
con los pares audio+texto reales de GLaDOS, para que el modelo aprenda
timbre **y** prosodia juntos, de punta a punta.

#### Preparación del dataset

- Transcripciones extraídas de `resource/subtitles_spanish.txt` de Portal 2
  (UTF-16LE, convertido a UTF-8).
- Los tokens `glados.<nombre_archivo>` cruzados automáticamente con los
  wavs extraídos: **1025 de 1112 archivos con transcripción exitosa (92%)**;
  el resto son efectos de sonido no hablados, exclusión correcta.
- Manifest final: `manifest.csv` (formato `archivo.wav|texto`).
- El doblaje resultó ser **español de España** (confirmado por vocabulario:
  "vosotros", "tío", etc.), no latino — importante para elegir el
  checkpoint base correcto.

#### Entorno

- `piper1-gpl` clonado, venv con Python 3.12.
- Checkpoint base: [`rhasspy/piper-checkpoints`](https://huggingface.co/datasets/rhasspy/piper-checkpoints),
  voz `es_ES/davefx/medium` (español de España, calidad "medium" para
  compatibilidad de fine-tuning).
- Instalación: `pip install -e '.[train]'` (el extra `[train]` es
  obligatorio, trae PyTorch/Lightning; instalar sin el extra deja el
  entorno sin lo necesario para entrenar).
- Extensión Cython de alineamiento monótono compilada con
  `./build_monotonic_align.sh` + `python3 setup.py build_ext --inplace`
  (requiere `scikit-build` y `cmake` como paquetes de Python, además de
  los binarios de sistema).

#### Entrenamiento

```bash
python3 -m piper.train fit \
  --data.voice_name "glados_es_tts" \
  --data.csv_path dataset-tools/manifest.csv \
  --data.audio_dir dataset-tools/glados_raw_es \
  --model.sample_rate 22050 \
  --data.espeak_voice "es" \
  --data.cache_dir piper-train/cache \
  --data.config_path piper-train/config_out.json \
  --data.batch_size 4 \
  --trainer.precision "bf16-mixed" \
  --ckpt_path checkpoints/<checkpoint_base>.ckpt
```

- **Batch size 4 + precisión `bf16-mixed`**: necesario para entrar en 8GB
  de VRAM (batch sizes mayores como 16 producen `CUDA out of memory`).
- Velocidad real: ~229 pasos/época, ~60-70 segundos/época, ~57-62
  épocas/hora en esta GPU.
- Regla general de la comunidad/documentación oficial de Piper: ~1000
  épocas adicionales para fine-tuning desde un checkpoint ya entrenado
  (criterio real: escuchar los checkpoints intermedios y/o observar que
  `loss_disc_all`/`loss_gen_all` se estabilicen en TensorBoard, más que
  un número fijo).
- Entrenamiento cortado manualmente (`Ctrl+C`, graceful shutdown) al
  considerar la calidad suficiente tras escuchar samples intermedios.

#### Exportación a ONNX

```bash
python3 export_onnx_wrapper.py \
  --checkpoint lightning_logs/version_X/checkpoints/last.ckpt \
  --output-file models-final/es_ES-glados-medium.onnx
```

`export_onnx_wrapper.py` es un wrapper mínimo alrededor de
`piper.train.export_onnx` (ver sección de problemas más abajo — necesario
por un cambio de comportamiento en versiones recientes de PyTorch).

---

## Problemas encontrados y soluciones

Documentado por si alguien repite este proceso con `piper1-gpl` y se
encuentra con lo mismo:

| Problema | Causa | Solución |
|---|---|---|
| `Weights only load failed` / `PosixPath not allowed` al cargar el checkpoint | PyTorch 2.6+ cambió el default de `torch.load` a `weights_only=True` | `torch.serialization.add_safe_globals([pathlib.PosixPath])` antes de cargar |
| `Subcommand 'fit' does not accept option 'model.sample_bytes'` (y luego `model.channels`) | El checkpoint de HuggingFace fue entrenado con una versión más vieja del código, con hiperparámetros que ya no existen en la clase de modelo actual | Script que limpia `checkpoint['hyper_parameters']`, quedándose solo con las claves que el modelo actual reconoce (lista blanca, no negra — hay ~60 claves obsoletas mezcladas) |
| `CUDA out of memory` al entrenar | `batch_size` demasiado alto para 8GB VRAM | Bajar a `batch_size 4` + `--trainer.precision bf16-mixed` |
| Crash al final de la primera época: `ModelCheckpoint(monitor='val_mos') could not find the monitored key` | El callback de calidad MOS (UTMOS) requiere `torchaudio`, no instalado; Lightning 2.6.5 lanza excepción dura (no solo warning) si la validación ya corrió | Wrapper que remueve ese callback opcional de `_DEFAULT_CALLBACKS`, dejando el callback de `val_mel` (que sí funciona) |
| Export a ONNX falla con `GuardOnDataDependentSymNode` | Desde PyTorch 2.9, `torch.onnx.export()` usa por defecto el exportador nuevo basado en `torch.export` ("dynamo"), que no puede trazar una aserción sobre valores de tensores que usa el código de VITS | Forzar `dynamo=False` (exportador legacy basado en TorchScript) interceptando la llamada a `torch.onnx.export` |
| `piper -m modelo.onnx` → `Unable to find voice` | Se corrió sin el venv activo, usando una instalación distinta de Piper (solo inferencia, sin código de entrenamiento) | Usar siempre el `piper` del venv del repo clonado, o invocar el python del venv por ruta completa |
| GLaDOS lee literalmente `*sonríe*` en voz alta | El modelo base (Ministral) tiende a insertar descripciones de acción entre asteriscos, estilo roleplay | Función `clean_for_speech()` que remueve texto entre asteriscos y símbolos markdown antes de sintetizar, + instrucción explícita en el `SYSTEM` prompt |
| `ollama launch claude` con modelos locales devuelve JSON crudo en vez de ejecutar la herramienta | Bug conocido de Ollama ([issue #15529](https://github.com/ollama/ollama/issues/15529)): la traducción al formato de tool-calling de la API de Anthropic falla con modelos locales (no con modelos cloud) | No es un problema de esta parte del proyecto — usar modelos `:cloud` para Claude Code, o construir el tool-calling propio directo contra la API nativa de Ollama (que sí funciona bien), como se hizo para el agente de este proyecto |

---

## El agente: herramientas y reconocimiento de voz {#el-agente}

`glados_agent.py` extiende el chat simple con:

**Tool calling nativo de Ollama** (no pasa por el bug de Claude Code, que
es específico de esa integración puntual):
- `get_weather(location)` — clima vía Open-Meteo (sin API key).
- `web_search(query)` — búsqueda vía DuckDuckGo (paquete `ddgs`, sin
  API key).
- `check_disk_usage()` / `check_memory_usage()` — comandos de sistema
  **fijos y predefinidos** (`df -h`, `free -h`). Deliberadamente no se
  implementó una herramienta de "ejecutar comando arbitrario": cada
  herramienta corre un comando exacto, sin que el modelo pueda construir
  o modificar el comando en sí. Esto evita por diseño (no solo por
  instrucción de prompt) que el modelo pueda crear, modificar o borrar
  archivos, incluso ante alucinaciones o intentos de inyección de
  instrucciones desde resultados de búsqueda web.
- `REQUIRES_CONFIRMATION`: set vacío por ahora (las herramientas actuales
  son de solo lectura); mecanismo listo para exigir confirmación manual
  si en el futuro se agregan herramientas con capacidad de modificar el
  sistema.

**Reconocimiento de voz (push-to-talk):**
- `faster-whisper` (modelo `small`, CPU, `int8`) — se eligió CPU en vez
  de GPU deliberadamente para no competir por VRAM con el LLM, que ya
  usa la mayor parte de los 8GB.
- Activación por teclado (Enter para empezar/terminar), no escucha
  continua — evita también el problema de que el micrófono capte el
  audio de los parlantes cuando GLaDOS está hablando.

---

## Demo de audio

Resultado del modelo de voz final (checkpoint `.onnx`, fine-tuning directo de
Piper sobre voz de GLaDOS — **solo** la voz clonada sintetizada, sin audio
original del juego). Escuchá la voz decir: *"Hola, soy GLaDOS, el modelo de
inteligencia artificial del juego de Valve conocido como Portal y Portal 2."*

<audio controls src="demo/glados-tts-demo.wav">
  Tu navegador no soporta la etiqueta <code>audio</code>. Descargá el archivo:
  <a href="demo/glados-tts-demo.wav">demo/glados-tts-demo.wav</a>
</audio>

> El repositorio **no** incluye audio extraído de Portal 2 (copyright de Valve).
> Solo se publica la voz sintetizada por el modelo entrenado, como ejemplo del
> resultado del pipeline.

---

## Estructura de carpetas

```
AI-Assistant/
├── modelfiles/
│   └── Modelfile.glados-ministral        # Personalidad del LLM
├── piper-train/                          # Instalación local piper1-gpl (no incluida en el repo)
│   └── venv/                             # Python 3.12 + torch cu130
├── scripts/                              # Wrappers y config de entrenamiento/export
│   ├── run_train.py                      # Wrapper de entrenamiento (fixes de compatibilidad)
│   ├── export_onnx_wrapper.py            # Wrapper de exportacion (fix dynamo)
│   ├── fix_checkpoint.py                 # Limpieza de hiperparametros obsoletos
│   └── config_out.json                   # Config del entrenamiento
├── models-final/
│   ├── es_ES-glados-medium.onnx          # Modelo de voz final
│   └── es_ES-glados-medium.onnx.json     # Config asociado
├── dataset-tools/
│   ├── glados_raw_es/                    # Audio extraido (NO subir a git)
│   ├── manifest.csv                      # Pares audio|texto
│   └── build_manifest.py
├── demo/
│   └── glados-tts-demo.wav               # Voz sintetizada de ejemplo (demo de audio)
├── rvc/                                  # Pipeline RVC (opcional, no usado en el pipeline final)
├── glados_agent.py                       # Agente principal (voz + tools)
└── README.md
```

---

## Cómo correrlo

```bash
# Dependencias del agente (dentro del venv de piper-train)
pip install requests sounddevice faster-whisper numpy ddgs

# Alias recomendado (Fish shell) en ~/.config/fish/config.fish:
alias glados="/ruta/a/piper-train/venv/bin/python3 /ruta/a/glados_agent.py"
```

```bash
glados
```

---

## Nota legal / assets con copyright {#nota-legal}

Este proyecto usa audio extraído de una copia legítima de Portal 2 con
fines personales y educativos. **El dataset de audio y el modelo de voz
entrenado no se distribuyen en este repositorio** — son derivados de la
interpretación de una actriz de doblaje profesional y propiedad
intelectual de Valve. Solo se publican el código, los scripts y la
metodología. Si clonás este repo para reproducir el proceso, necesitás tu
propia copia legítima del juego para extraer el audio.

`.gitignore` sugerido:
```
dataset-tools/glados_raw_es/
dataset-tools/glados_raw/
models-final/*.onnx
piper-train/checkpoints/
piper-train/lightning_logs/
piper-train/cache/
rvc/logs/
*.pth
*.ckpt
```

---

## Créditos

- [Portal 2](https://store.steampowered.com/app/620/Portal_2/) — Valve.
  GLaDOS y su interpretación de voz son propiedad de Valve Corporation.
- [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) — motor de TTS
  usado para el fine-tuning.
- [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints) —
  checkpoint base (`es_ES/davefx/medium`).
- [Applio](https://github.com/IAHispano/Applio) — fork de RVC usado en
  la etapa experimental de conversión de timbre.
- [Ollama](https://ollama.com/) — runtime del LLM.
