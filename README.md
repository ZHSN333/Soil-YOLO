# Soil-YOLO
Lightweight Texture-Aware Detection of Geological Materials in Foundation Pit Images
## Contents

- `soil_yolo/modules.py`: texture enhancement, ECA and fusion modules.
- `soil_yolo/loss.py`, `model.py`, `trainer.py`, `register.py`: loss, model, trainer and custom module registration.
- `soil_yolo/augment.py`, `config.py`, `pipeline.py`: augmentation, loss settings and execution pipeline.
- `main.py`: training, resume, inference and curve plotting; `voc2yolo.py` converts VOC annotations.

## Environment

Install Python 3.10 or later and PyTorch/torchvision appropriate for your operating system and CUDA, then install the remaining dependencies:

```bas
pip install ultralytics albumentations opencv-python matplotlib pandas pillow pyyaml
```

## Inference and continuation

```bash
python main.py val --weights path/to/best.pt --source path/to/images --no-metrics --device 0
python main.py resume --ckpt path/to/last.pt
python main.py plot --run path/to/training_run
python main.py --help
```
