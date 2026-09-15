import vpk
import os

VPK_PATH = "/mnt/games/Steam/steamapps/common/Portal 2/portal2_spanish/pak01_dir.vpk"
OUTPUT_DIR = "/mnt/games/AI-Assistant/dataset-tools/glados_raw_es"

os.makedirs(OUTPUT_DIR, exist_ok=True)

pak = vpk.open(VPK_PATH)

glados_files = [p for p in pak if p.lower().startswith("sound/vo/glados/")]

print(f"Extrayendo {len(glados_files)} archivos...")

extracted = 0
failed = 0

for path in glados_files:
    try:
        data = pak.get_file(path).read()
        filename = os.path.basename(path)
        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, "wb") as f:
            f.write(data)
        extracted += 1
    except Exception as e:
        print(f"Error con {path}: {e}")
        failed += 1

print(f"\nExtraídos: {extracted}")
print(f"Fallidos: {failed}")
print(f"Carpeta destino: {OUTPUT_DIR}")
