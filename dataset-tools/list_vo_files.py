import vpk
import sys

VPK_PATH = "/mnt/games/Steam/steamapps/common/Portal 2/portal2_spanish/pak01_dir.vpk"

pak = vpk.open(VPK_PATH)

vo_paths = [p for p in pak if "sound/vo" in p.lower()]

print(f"Total de archivos en el VPK: {sum(1 for _ in pak)}")
print(f"Archivos bajo sound/vo: {len(vo_paths)}")
print("\nPrimeros 40 ejemplos:")
for p in vo_paths[:40]:
    print(p)

# Guarda la lista completa para revisarla con calma
with open("/mnt/games/AI-Assistant/dataset-tools/vo_file_list.txt", "w") as f:
    f.write("\n".join(vo_paths))

print(f"\nLista completa guardada en vo_file_list.txt ({len(vo_paths)} rutas)")
