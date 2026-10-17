"""nightgan: a latent-space transformer GAN for a single GPU."""

__version__ = "0.1.0"


def load(path, device=None):
    """Loads a trained model (`model.safetensors` or a run folder) as a `Pipeline`."""
    from .pipeline import Pipeline
    return Pipeline.load(path, device=device)
