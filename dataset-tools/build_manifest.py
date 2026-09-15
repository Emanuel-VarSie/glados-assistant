import re
import os
import csv
import glob

SUBTITLES_PATH = "/mnt/games/AI-Assistant/dataset-tools/captions/subtitles_spanish_utf8.txt"
AUDIO_DIR = "/mnt/games/AI-Assistant/dataset-tools/glados_raw_es"
OUTPUT_MANIFEST = "/mnt/games/AI-Assistant/dataset-tools/manifest.csv"
OUTPUT_UNMATCHED = "/mnt/games/AI-Assistant/dataset-tools/unmatched_files.txt"

# --- 1. Parsear el archivo de subtítulos ---
# Formato: "token"    "<clr:r,g,b>GLaDOS: texto..."
# Nos interesan solo los tokens que empiezan con glados. (case-sensitive, en minúsculas)
# y que NO empiecen con [english] (esas son la versión en inglés, se ignoran)

line_pattern = re.compile(r'^"(glados\.[^"]+)"\s+"(.*)"\s*$')
color_tag_pattern = re.compile(r'<clr:\d+,\d+,\d+>')
speaker_prefix_pattern = re.compile(r'^GLaDOS:\s*', re.IGNORECASE)

token_to_text = {}

with open(SUBTITLES_PATH, "r", encoding="utf-8-sig") as f:
    for line in f:
        line = line.strip()
        if line.startswith('"[english]'):
            continue
        match = line_pattern.match(line)
        if not match:
            continue
        token, raw_text = match.group(1), match.group(2)

        # Quitar prefijo "glados."
        file_stem = token[len("glados."):]

        # Limpiar texto: quitar tags de color y el prefijo "GLaDOS: "
        clean_text = color_tag_pattern.sub("", raw_text)
        clean_text = speaker_prefix_pattern.sub("", clean_text).strip()

        # Ignorar entradas vacías o que sean solo efectos [algo]
        if not clean_text or (clean_text.startswith("[") and clean_text.endswith("]")):
            continue

        token_to_text[file_stem] = clean_text

print(f"Transcripciones parseadas: {len(token_to_text)}")

# --- 2. Cruzar con los archivos de audio extraídos ---
audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
print(f"Archivos de audio encontrados: {len(audio_files)}")

matched = []
unmatched = []

for audio_path in audio_files:
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    if stem in token_to_text:
        matched.append((os.path.basename(audio_path), token_to_text[stem]))
    else:
        unmatched.append(os.path.basename(audio_path))

print(f"\nCoincidencias encontradas: {len(matched)}")
print(f"Sin coincidencia: {len(unmatched)}")

# --- 3. Escribir el manifest (formato: filename|texto) ---
with open(OUTPUT_MANIFEST, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="|")
    for filename, text in matched:
        writer.writerow([filename, text])

print(f"\nManifest guardado en: {OUTPUT_MANIFEST}")

# --- 4. Guardar lista de archivos sin transcripción, para revisar ---
with open(OUTPUT_UNMATCHED, "w", encoding="utf-8") as f:
    f.write("\n".join(unmatched))

print(f"Archivos sin match guardados en: {OUTPUT_UNMATCHED}")

# --- 5. Muestra algunos ejemplos ---
print("\nPrimeros 10 ejemplos del manifest:")
for filename, text in matched[:10]:
    print(f"  {filename} -> {text}")

if unmatched:
    print(f"\nPrimeros 10 archivos SIN transcripción (revisar patrón):")
    for f_name in unmatched[:10]:
        print(f"  {f_name}")
