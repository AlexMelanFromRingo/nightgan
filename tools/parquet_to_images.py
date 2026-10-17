"""Extracts the images of Hugging Face parquet datasets (an `image` column with encoded bytes) into a folder.

    python tools/parquet_to_images.py ~/datasets/hf/merkol__ffhq-256/data ~/datasets/ffhq256
    python tools/parquet_to_images.py ~/datasets/hf/huggan__AFHQv2/data ~/datasets/afhqv2 --label-column label

With `--label-column`, images go to one subfolder per label (class names from the dataset features if present),
which `nightgan prepare --labels dirs` reads as classes. The encoded bytes are written as they are, not re-encoded.
Needs pyarrow (`pip install pyarrow`).
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def extension(data: bytes) -> str:
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return '.png'
    if data[:3] == b'\xff\xd8\xff':
        return '.jpg'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return '.webp'
    return '.img'


def class_names(files, column, src: Path):
    """Label names from the parquet metadata, or from a dataset_infos.json next to the data (Hugging Face)."""
    meta = pq.ParquetFile(files[0]).schema_arrow.metadata or {}
    info = meta.get(b'huggingface')
    if info:
        feature = json.loads(info).get('info', {}).get('features', {}).get(column, {})
        if feature.get('names'):
            return feature['names']
    for folder in (src, src.parent):
        infos = folder / 'dataset_infos.json'
        if infos.exists():
            for config in json.loads(infos.read_text()).values():
                names = config.get('features', {}).get(column, {}).get('names')
                if names:
                    return names
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('src', help='folder with .parquet files (searched recursively)')
    p.add_argument('dst', help='output folder')
    p.add_argument('--image-column', default='image')
    p.add_argument('--label-column', help='write one subfolder per value of this column')
    args = p.parse_args()

    files = sorted(Path(args.src).expanduser().rglob('*.parquet'))
    if not files:
        raise SystemExit(f'no parquet files in {args.src}')
    names = class_names(files, args.label_column, Path(args.src).expanduser()) if args.label_column else None
    dst = Path(args.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    columns = [args.image_column] + ([args.label_column] if args.label_column else [])
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=columns):
            rows = batch.to_pydict()
            for i, image in enumerate(rows[args.image_column]):
                data = image['bytes']
                folder = dst
                if args.label_column:
                    label = rows[args.label_column][i]
                    folder = dst / (names[label] if names else str(label))
                    folder.mkdir(exist_ok=True)
                (folder / f'{n:06d}{extension(data)}').write_bytes(data)
                n += 1
        print(f'{f.name}: {n} images so far', flush=True)
    print(f'{n} images in {dst}')


if __name__ == '__main__':
    main()
