import functools

import torch

_original_export = torch.onnx.export


@functools.wraps(_original_export)
def _export_dynamo_false(*args, **kwargs):
    kwargs.setdefault("dynamo", False)
    return _original_export(*args, **kwargs)


torch.onnx.export = _export_dynamo_false

from piper.train.export_onnx import main

if __name__ == "__main__":
    main()
