# nightgan

A GAN you can train on one consumer GPU, built from the strongest published GAN methods of 2024–2026:
transformer generator and discriminator in the latent space of a frozen autoencoder ([GAT](https://arxiv.org/abs/2509.24935)),
a scale-wise discriminator with stage consistency ([CAT](https://arxiv.org/abs/2605.26449)),
the relativistic pairing loss with R1 + R2 ([R3GAN](https://arxiv.org/abs/2501.05441)) computed without double
backpropagation, and representation alignment of the discriminator to DINOv2 ([REPA](https://arxiv.org/abs/2410.06940)).

- Pure PyTorch: no custom CUDA kernels, no Docker, `pip install` and go.
- Three commands: `prepare` a folder of images, `train`, `generate`.
- Resumes where it stopped (Ctrl+C, crash, reboot), picks the batch size that fits your GPU, evaluates FID / KID
  while training, exports a small `.safetensors` model.
- Unconditional or class-conditional (one subfolder per class), 128 / 256 / 512 px.

See [docs/how-it-works.md](docs/how-it-works.md) for the method and [docs/results.md](docs/results.md) for measured
results.

## Install

Python 3.10 or newer and a PyTorch build with CUDA (see [pytorch.org](https://pytorch.org/get-started/locally/) if
the default wheel of your platform has no CUDA):

```bash
git clone https://github.com/AlexMelanFromRingo/nightgan && cd nightgan
pip install -e .            # add [video] for .mp4 interpolations
```

The first run downloads the Stable Diffusion VAE (330 MB, Hugging Face), DINOv2 ViT-B/14 (350 MB, torch.hub) and
the FID Inception network (100 MB); afterwards everything works offline.

## Quick start

```bash
# 1. encode the images once (any folder, searched recursively; jpg / png / webp / ...)
nightgan prepare ~/my_images data/mine --resolution 256

# 2. train; running the same command again resumes
nightgan train data/mine runs/mine --preset S --time 10h

# 3. use it
nightgan generate runs/mine out/ --seeds 0-63 --grid
nightgan interpolate runs/mine walk.gif --seeds 1,7,42 --frames 60
nightgan eval runs/mine data/mine --metrics fid,kid
```

Training writes to the run folder: `samples/` (a grid of fixed seeds every 100k images), `log.jsonl` (losses,
speed, FID), `checkpoints/` (full training state, every 30 minutes and at the end), `model.safetensors` (the
generator, latest) and `model-best.safetensors` (best FID so far).

Class-conditional: put each class in its own subfolder and prepare with `--labels dirs`, then
`nightgan generate runs/mine out/ --class 2`.

## Presets and speed

Measured on an RTX 4080 SUPER (16 GB, WSL2), batch 256, bf16, `torch.compile`:

| Preset | Generator / discriminator | Images / s (with REPA) | GPU memory | Images in 10 h |
|---|---|---|---|---|
| `S` | 29M / 37M | 143 | 7.2 GB | ~5.1 M |
| `B` | 117M / 102M | 58 | 9.7 GB | ~2.1 M |

The resolution does not change the speed much: 128, 256 and 512 px all use 16 x 16 tokens by default (latent patch
1, 2 and 4); `--patch` can be lowered for more tokens at a higher cost. Without REPA (`--repa-weight 0`) training is
~12 % faster.

## Training options

| Option | Default | Meaning |
|---|---|---|
| `--preset` | `S` | model size; `--dim --depth --heads --d-dim --d-depth --d-heads --z-dim --patch` override it |
| `--time` / `--kimg` | until stopped | training length (`10h`, `90m`, `2d`; total over all sessions) or thousands of images; the learning rate decays over the last third |
| `--batch` | 256 | images per optimizer step |
| `--micro-batch` | 0 (auto) | images per forward pass; the largest that fits ~85 % of the free GPU memory, the rest is gradient accumulation |
| `--lr-g`, `--lr-d` | `2e-4 * 768 / width` | learning rates (width-aware, from GAT) |
| `--gamma` | 0.1 | weight of the R1 + R2 gradient penalties |
| `--gp-sigma` | 0.03 | noise of the finite-difference gradient penalty |
| `--cons-weight` | 0.1 | stage consistency (CAT) |
| `--repa-weight`, `--repa-encoder` | 1, `dinov2_vitb14` | representation alignment of the discriminator; `--repa-weight 0` disables it |
| `--aug-policy`, `--aug-p` | `color,translation,cutout`, 1 | differentiable augmentation of the latents |
| `--ema-halflife-kimg` | 350 | half-life of the generator's moving average, in thousands of images |
| `--eval-every-kimg`, `--eval-num` | 500, 10000 | FID / KID during training (0 disables) |
| `--sample-every-kimg`, `--snapshot-minutes` | 100, 30 | sample grids and snapshots |
| `--no-compile`, `--checkpointing` | | eager mode (Windows without triton) / gradient checkpointing (less memory) |
| `--force-config` | | start over in a run folder with a different model (old snapshots are moved aside) |

`nightgan <command> --help` lists everything.

## Datasets

`nightgan prepare` center-crops every image to a square, resizes it (Lanczos), applies EXIF rotation, puts transparent
images on white, converts palette / grayscale / CMYK / 16-bit images, and skips files it cannot read (they are listed
at the end). Each image is encoded together with its mirror image. The dataset folder holds the latents, 224 px copies
for REPA and the reference statistics for FID / FD-DINOv2: about 180 KB per image plus up to ~170 MB of statistics.

Small datasets (a few hundred to a few thousand images) work: the latents are augmented (DiffAugment) and the
discriminator is regularized by REPA. Expect the best results with 10k+ images.

## Python

```python
import nightgan

pipe = nightgan.load('runs/mine/model.safetensors')     # or the run folder
images = pipe.generate(seeds=range(8), truncation=0.8)   # list of PIL images
frames = pipe.interpolate([1, 7, 42], frames=30)
```

## Troubleshooting

- **Training is several times slower than the table on Windows / WSL2.** The GPU memory is full and the driver moves
  memory to the system RAM instead of failing. Close other GPU programs or set `--micro-batch` lower; the automatic
  choice leaves 15 % of the free memory unused.
- **Windows without triton:** use `--no-compile` (~20 % slower).
- **"non-finite loss … step skipped":** a single skipped step is harmless; three in a row stop the run with a pointer
  to the last snapshot. Lower `--lr-g` / `--lr-d` or raise `--gamma`, then run the same command to resume.
- **Offline:** once the weights are downloaded nothing else is needed. The datasets and the run folders can be moved.

## Tests

```bash
pip install -e .[dev] && pytest            # offline, ~40 s
pytest -m slow                             # with the pretrained networks (downloads them)
```

## Credits and licenses

nightgan is MIT licensed. It uses the Stable Diffusion VAE `stabilityai/sd-vae-ft-mse` (MIT), DINOv2 (Apache-2.0) and
the FID InceptionV3 port of pytorch-fid (Apache-2.0). The method combines GAT (Hyun, Lee, Heo, ICML 2026), CAT
(arXiv 2605.26449), R3GAN (Huang et al., NeurIPS 2024), REPA (Yu et al., ICLR 2025) and DiffAugment (Zhao et al., 2020);
the code is written from the papers.
