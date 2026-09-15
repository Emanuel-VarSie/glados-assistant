import pathlib
import sys

import torch.serialization

torch.serialization.add_safe_globals([pathlib.PosixPath])

import piper.train.__main__ as piper_main

# El callback val_mos depende de un predictor de calidad (UTMOS) que
# necesita torchaudio, no instalado. Si val_mos nunca se loguea,
# Lightning 2.6.5 tira error duro (no solo warning) una vez que la
# validacion ya corrio. Sacamos ese callback opcional; el de val_mel
# sigue funcionando normal y es el que de verdad importa.
piper_main._DEFAULT_CALLBACKS = [
    cb for cb in piper_main._DEFAULT_CALLBACKS
    if getattr(cb, "monitor", None) != "val_mos"
]

if __name__ == "__main__":
    piper_main.main()
