# TASED-Net (vendored model definition)

- Upstream: https://github.com/MichiganCOG/TASED-Net
- Upstream commit: `0cf47b73cf055bfc64dc132a85276cca67308226` (2023-07-19)
- File: `model.py`, copied verbatim (SHA-256 `caa050e71e3fdecfa3e8230ab6e23b0215ad049dd8189e5c73ad3f2c1caea884`).
  It imports only `torch` / `torch.nn`. No other upstream file (training, data
  loading, `run_example.py`) is vendored or executed; the inference protocol of
  `run_example.py` is re-implemented in `autogaze/human_gaze/saliency_baseline.py`.
- License: MIT, Copyright (c) 2019 Kyle Min (`LICENSE`, copied verbatim).
- Paper: Min & Corso, "TASED-Net: Temporally-Aggregating Spatial Encoder-Decoder
  Network for Video Saliency Detection", ICCV 2019, arXiv:1908.05786.
- Weights (not tracked): `TASED_updated.pt` from the README link
  https://drive.google.com/file/d/1y4KSTm-e7kP84k0IyVI-rRtmYZkrC-wL ,
  SHA-256 `8be860c8faf41674f5f2e23ea8e1cef2290ecc7e4f6ed0a75d61c0e117588afd`,
  stored under `/storage/user/latn/models/sb1/`; loaded with
  `torch.load(weights_only=True)` (plain state dict of 513 tensors).
