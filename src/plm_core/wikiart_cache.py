"""One-time cache of a WikiArt exemplar subset for the ARH item generator.

Streams huggan/wikiart (81k images total) rather than downloading it whole, and stops
once each selected style has enough exemplars. Ground truth = the dataset's own curated
style/artist/genre ClassLabels — never LLM-asserted, matching the project's oracle policy.

Run directly: `python -m src.plm_core.wikiart_cache`. Idempotent — skips styles that
already have enough cached images.
"""

import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image

CACHE_DIR = Path("data/wikiart_cache")
MANIFEST_PATH = CACHE_DIR / "manifest.json"
PER_STYLE = 60
MAX_SIDE = 900  # resize cap so cached files stay small; crops are taken after this resize

# Five visually distinct styles (avoids confusable near-neighbors like Early/High
# Renaissance for this first working generator).
STYLES = ["Baroque", "Impressionism", "Cubism", "Ukiyo_e", "Pop_Art"]


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def build_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    for style in STYLES:
        manifest.setdefault(style, [])

    remaining = {s: PER_STYLE - len(manifest[s]) for s in STYLES}
    if all(n <= 0 for n in remaining.values()):
        print("all styles already cached — nothing to do")
        return

    print(f"need: { {s: n for s, n in remaining.items() if n > 0} }")
    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    style_names = ds.features["style"].names
    artist_names = ds.features["artist"].names

    scanned = 0
    for ex in ds:
        scanned += 1
        style = style_names[ex["style"]]
        if style in STYLES and remaining[style] > 0:
            idx = len(manifest[style])
            img = ex["image"].convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            style_dir = CACHE_DIR / style
            style_dir.mkdir(exist_ok=True)
            out_path = style_dir / f"{idx:03d}.jpg"
            img.save(out_path, "JPEG", quality=88)
            manifest[style].append(
                {"path": str(out_path), "artist": artist_names[ex["artist"]], "style": style}
            )
            remaining[style] -= 1
            print(f"[{style}] {idx + 1}/{PER_STYLE} cached (scanned {scanned})", flush=True)
            _save_manifest(manifest)  # incremental — survives a crash/kill mid-scan

        if all(n <= 0 for n in remaining.values()):
            break
        if scanned % 2000 == 0:
            print(f"...scanned {scanned}, still need {[s for s, n in remaining.items() if n > 0]}", flush=True)

    _save_manifest(manifest)
    print(f"done — manifest at {MANIFEST_PATH}")


if __name__ == "__main__":
    build_cache()
