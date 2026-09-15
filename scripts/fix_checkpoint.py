import sys
import pathlib

import torch
import torch.serialization

torch.serialization.add_safe_globals([pathlib.PosixPath])

ALLOWED_MODEL_KEYS = {
    "sample_rate", "num_speakers", "resblock", "resblock_kernel_sizes",
    "resblock_dilation_sizes", "upsample_rates", "upsample_initial_channel",
    "upsample_kernel_sizes", "filter_length", "hop_length", "win_length",
    "mel_channels", "mel_fmin", "mel_fmax", "inter_channels",
    "hidden_channels", "filter_channels", "n_heads", "n_layers",
    "kernel_size", "p_dropout", "n_layers_q", "use_spectral_norm",
    "gin_channels", "use_sdp", "segment_size", "use_mrd", "learning_rate",
    "learning_rate_d", "betas", "betas_d", "eps", "lr_decay", "lr_decay_d",
    "init_lr_ratio", "warmup_epochs", "c_mel", "c_kl", "grad_clip",
    "vocoder_warmstart_ckpt", "warmstart_ckpt", "mos_metric", "dataset",
}


def main():
    if len(sys.argv) != 3:
        print("Uso: python3 fix_checkpoint.py <entrada.ckpt> <salida.ckpt>")
        sys.exit(1)

    src_path, dst_path = sys.argv[1], sys.argv[2]

    print(f"Cargando checkpoint: {src_path}")
    ckpt = torch.load(src_path, weights_only=False, map_location="cpu")

    hparams = ckpt.get("hyper_parameters", {})
    print(f"Claves encontradas en hyper_parameters ({len(hparams)}):")
    for k in sorted(hparams.keys()):
        print(f"  - {k}")

    all_keys = set(hparams.keys())
    to_remove = sorted(all_keys - ALLOWED_MODEL_KEYS)

    for key in to_remove:
        del hparams[key]

    if to_remove:
        print(f"\nEliminadas {len(to_remove)} claves no reconocidas por el modelo actual:")
        for k in to_remove:
            print(f"  - {k}")
    else:
        print("\nNo se encontraron claves para eliminar.")

    print(f"\nClaves que quedan ({len(hparams)}): {sorted(hparams.keys())}")

    print(f"\nGuardando checkpoint limpio en: {dst_path}")
    torch.save(ckpt, dst_path)
    print("Listo.")


if __name__ == "__main__":
    main()
