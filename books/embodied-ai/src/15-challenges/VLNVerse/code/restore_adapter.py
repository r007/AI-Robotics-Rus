"""Restore a byte-identical adapter from SHA-256 verified release chunks."""
import argparse
import hashlib
import json
from pathlib import Path


def restore(root):
    manifest = json.loads((root/'weight_manifest.json').read_text())
    target = root/'trainable_params.bin'
    h = hashlib.sha256()
    with target.open('xb') as output:
        for part in manifest['parts']:
            relative = Path(part['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid chunk path')
            data = (root/relative).read_bytes()
            if len(data) != part['size'] or hashlib.sha256(data).hexdigest() != part['sha256']:
                raise ValueError(f'Chunk checksum mismatch: {relative}')
            output.write(data)
            h.update(data)
    if target.stat().st_size != manifest['size'] or h.hexdigest() != manifest['sha256']:
        raise ValueError('Restored weight checksum mismatch')
    print('Verified:', target.name, h.hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    restore(parser.parse_args().directory)
