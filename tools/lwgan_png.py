# lightweight-gan saves generated images as JPEG (quality 75), which raises FID; this runs its CLI with PNG output.
# Usage: <python of the lightweight-gan environment> tools/lwgan_png.py <lightweight_gan arguments>
import sys
from lightweight_gan.lightweight_gan import Trainer
from lightweight_gan.cli import main
Trainer.image_extension = property(lambda self: 'png')
sys.exit(main())
