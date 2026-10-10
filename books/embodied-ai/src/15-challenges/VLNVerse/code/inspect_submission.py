"""Validate an Overall trajectory archive without ground-truth scoring."""

import argparse
import gzip
import hashlib
import json
import math
import zipfile
from pathlib import Path


def inspect(path):
    expected = {"coarse_challenge.json.gz", "fine_challenge.json.gz"}
    summary = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "tracks": {}}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 2 or set(names) != expected:
            raise ValueError("Expected exactly two root-level challenge files")
        for name in sorted(expected):
            episodes = json.loads(gzip.decompress(archive.read(name)))["episodes"]
            ids = [str(episode["episode_id"]) for episode in episodes]
            if len(episodes) != 150 or len(set(ids)) != 150:
                raise ValueError(f"{name}: expected 150 unique episodes")
            for episode in episodes:
                points = episode["reference_path"]
                if not points:
                    raise ValueError(f"{name}: empty trajectory")
                for point in points:
                    if not isinstance(point, list) or len(point) != 3:
                        raise ValueError(f"{name}: expected XYZ coordinate")
                    if not all(type(v) in (int, float) and math.isfinite(v) for v in point):
                        raise ValueError(f"{name}: invalid coordinate")
            summary["tracks"][name] = {"episodes": len(episodes), "unique": len(set(ids)), "empty": 0}
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.archive), indent=2))
