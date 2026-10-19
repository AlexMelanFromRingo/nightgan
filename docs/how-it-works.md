# How nightgan works

nightgan is a recipe assembled from recent GAN papers and adapted to a single 16 GB GPU and to datasets of a few
thousand to a hundred thousand images. This page describes what it does and why; the code follows it closely.

## Latent space

Images are encoded once, by `nightgan prepare`, with the frozen Stable Diffusion VAE (`sd-vae-ft-mse`, 8x
downsampling, 4 channels): a 256 x 256 image becomes a 4 x 32 x 32 latent. The GAN generates latents, the VAE decoder
turns them into images. This is what makes the training cheap: the networks see 16 x 16 tokens instead of 65 536
pixels, and no image has to be decoded during training.

The VAE gives a Gaussian per latent (mean and log-variance); both are stored, for every image and its mirror image,
and a new latent is drawn from it at every step (a free, mild augmentation, as in GAT). Latents are standardized per
channel with statistics of the dataset.

`--autoencoder pixel` skips the VAE and trains on the pixels; it is meant for small images (32 - 64 px) and tests.

The VAE sets a ceiling: a generator can at best reproduce the latents of the dataset, and decoding them loses some
detail. `nightgan prepare` measures it (FID / KID of 10 000 dataset images decoded from sampled latents, stored as
`reconstruction` in `meta.json`; `nightgan eval DATA --reconstruction` repeats it). On FFHQ 256 it is FID-10k 2.35,
on the 833 Pokemon images FID-5k 8.34; the trained models are far above it (11.2 and 117.4), so there the GAN, not the
VAE, limits the quality.

## Generator

A transformer on a 16 x 16 grid of tokens (`patch = latent size / 16`), following GAT:

- the input tokens are fixed 2D sine-cosine position embeddings; the latent `z` (64 numbers) and the class enter only
  through a mapping network (two layers, learning rate x0.01 as in StyleGAN) whose output `c` modulates every block;
- each block: RMSNorm → scale by `1 + s(c)` → self-attention (qk-normalization, 2D rotary embeddings) → gate `g(c)`,
  then the same around a SwiGLU MLP. The modulation layers start small (gain 0.1);
- four output heads, after each quarter of the blocks, are added up (output skip connections). The partial sums are
  the four *stages*; the last one is the generated latent.

Truncation (`--truncation` below 1) moves `c` towards the running average of the class, as in StyleGAN.

## Discriminator

A transformer that judges a latent at four resolutions (CAT): stage k of the generator is area-downsampled to
1/2^(3-k) of the full size (4, 8, 16, 32 for a 256 px image), real latents are downsampled the same way. All scales
share the patch embedding and the blocks, each scale has its own [cls] token and scale embedding, and attention stays
inside a scale (four fused attention calls instead of a masked one). Rotary embeddings use the coordinates of the
finest grid, so a coarse token sits at the centre of the fine tokens it covers. Each scale's [cls] token gives a logit;
with classes, the logit is a projection onto a class embedding (projection discriminator). LayerScale (0.1) keeps the
early training of the deep discriminator smooth.

## Losses

- **Relativistic pairing (RpGAN):** for a real latent `r` and a fake `f`, the discriminator minimizes
  `softplus(D(f) - D(r))` and the generator `softplus(D(r) - D(f))`, per scale. R3GAN shows that with R1 and R2 this
  loss has local convergence guarantees and needs none of the usual GAN tricks.
- **R1 and R2 without double backpropagation:** the squared gradient norm of D at real (R1) and fake (R2) inputs is
  estimated by a finite difference, `((D(x + e) - D(x)) / sigma)^2` with `e ~ N(0, sigma^2)` (GAT). This keeps fused
  attention usable and costs one extra forward pass on a quarter of the batch. The noisy copies go through the same
  discriminator call as the clean ones. In bfloat16 the estimate is biased upwards for small `sigma` (+26 % at
  0.01, x4 at 0.003, measured), hence the default `sigma = 0.03`.
- **Stage consistency (CAT):** the mean squared distance of the first three stages to the final output, weighted
  1/3, 1/2, 1, times 0.1. Because the stages are partial sums, this penalizes what the later heads add. The final
  output must not be detached: with a stop-gradient the early heads and the target move together, the distance never
  shrinks and the outputs drift without bound (found the hard way: samples collapsed to one texture in 120 steps).
- **REPA:** the discriminator's last-layer tokens of the finest scale (and its [cls]) are projected by a 3-layer MLP
  and aligned, by cosine similarity, to DINOv2 ViT-B/14 features of the same real image (GAT). The real image gets
  the same mirroring and translation as its latent, so the targets stay aligned with what D sees. This gives the
  discriminator good features early, which helps most on small datasets.

## Augmentation

DiffAugment on the standardized latents (GAT): color (brightness, saturation, contrast), translation by up to 1/8 of
the size, and cutout of a half-size square, each with probability 1, the same parameters for all stages of a fake.
Mirroring is done by the data (both orientations are encoded).

## Optimization

The default is AdamW with beta1 = 0, as in R3GAN, GAT and StyleGAN: momentum makes the generator and the
discriminator oscillate around each other. `--optimizer adabelief` is available for comparison; it needs beta1 > 0
(0.5 by default), because its second moment is the variance of the gradient around its running mean, which is zero
without momentum.


Details: AdamW with β = (0, 0.99) and no weight decay, learning rate `2e-4 * 768 / width` (width-aware, GAT; 4e-4 for the S
preset), gradient clipping at norm 1, batch 256 with gradient accumulation over micro-batches that fit the GPU, bf16
autocast with normalization layers kept in bf16 (autocast would return float32 and double the activation memory), and
`torch.compile` on the transformer blocks (+25 % speed). The generator used for sampling and metrics is an exponential
moving average with a half-life of 350k images (shorter early in training). With a known training length the learning
rate decays (cosine, to 25 %) over the last third.

## Sources

- GAT: Hyun, Lee, Heo. *Scalable GANs with Transformers*. ICML 2026. [arXiv 2509.24935](https://arxiv.org/abs/2509.24935)
- CAT: *Cross-scale Aligned Supervision for Training GANs*. 2026. [arXiv 2605.26449](https://arxiv.org/abs/2605.26449)
- R3GAN: Huang, Gokaslan, Kuleshov, Tompkin. *The GAN is dead; long live the GAN!* NeurIPS 2024. [arXiv 2501.05441](https://arxiv.org/abs/2501.05441)
- REPA: Yu et al. *Representation Alignment for Generation*. ICLR 2025. [arXiv 2410.06940](https://arxiv.org/abs/2410.06940)
- DiffAugment: Zhao et al. *Differentiable Augmentation for Data-Efficient GAN Training*. NeurIPS 2020. [arXiv 2006.10738](https://arxiv.org/abs/2006.10738)
- FD-DINOv2: Stein et al. *Exposing flaws of generative model evaluation metrics*. NeurIPS 2023. [arXiv 2306.04675](https://arxiv.org/abs/2306.04675)
