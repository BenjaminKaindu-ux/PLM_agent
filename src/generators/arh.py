"""ARH style-classification item generator — huggan/wikiart is the ground-truth oracle.

Every answer key is the dataset's own curated `style` ClassLabel (never LLM-asserted).
Per ARCHITECTURE.md, items present a cropped detail/fragment rather than the full
artwork, since the perceptual skill being trained is fast style recognition from
textural/compositional cues, not whole-composition memorization.
"""

import json
import random
from pathlib import Path

from PIL import Image

MANIFEST_PATH = Path("data/wikiart_cache/manifest.json")

# One-line, hand-authored descriptors of well-established stylistic cues (not
# LLM-generated at runtime) — same authorship model as MATH223's _FEEDBACK dict.
_STYLE_INFO = {
    "Baroque": {
        "display": "Baroque",
        "cue": "Dramatic chiaroscuro (strong light/dark contrast), dynamic diagonal composition, rich dark backgrounds.",
    },
    "Impressionism": {
        "display": "Impressionism",
        "cue": "Loose, visible brushstrokes; emphasis on fleeting light and color over precise outlines.",
    },
    "Cubism": {
        "display": "Cubism",
        "cue": "Fragmented, geometric planes showing multiple viewpoints of a subject at once.",
    },
    "Ukiyo_e": {
        "display": "Ukiyo-e",
        "cue": "Japanese woodblock printing: flat areas of color, bold outlines, no Western-style shading.",
    },
    "Pop_Art": {
        "display": "Pop Art",
        "cue": "Bold flat color blocks, mass-media/commercial imagery, high-contrast graphic style.",
    },
}

CATEGORIES = {style: {"rt_threshold_s": 10.0} for style in _STYLE_INFO}


def _get_manifest() -> dict:
    """Re-read every call (no caching): the wikiart_cache job may still be filling in
    styles in the background, and picking those up without a restart is the point."""
    if not MANIFEST_PATH.exists():
        raise RuntimeError(f"{MANIFEST_PATH} not found — run `python -m src.plm_core.wikiart_cache` first")
    return json.loads(MANIFEST_PATH.read_text())


def active_categories() -> list[str]:
    """Styles with at least one cached exemplar — grows automatically as
    wikiart_cache.py fills in the rest, e.g. currently 3 of 5 (Pop_Art/Ukiyo_e pending)."""
    manifest = _get_manifest()
    return [s for s in _STYLE_INFO if manifest.get(s)]


def _crop_detail(img: Image.Image, rng: random.Random, frac_range=(0.4, 0.6)) -> Image.Image:
    """Crop a random square 'detail' covering frac of the shorter side, then upscale
    to a consistent display size — forces reliance on local texture/color cues."""
    w, h = img.size
    side = int(min(w, h) * rng.uniform(*frac_range))
    x = rng.randint(0, w - side)
    y = rng.randint(0, h - side)
    crop = img.crop((x, y, x + side, y + side))
    return crop.resize((480, 480), Image.LANCZOS)


def make_item(category: str, rng: random.Random | None = None, difficulty: int = 1) -> dict:
    """Generate one schema-compliant ARH FieldSense-style item: crop a detail from a
    real WikiArt exemplar and ask the student to classify its style among the active
    category set. Ground truth is the dataset's own style label."""
    rng = rng or random.Random()
    manifest = _get_manifest()
    if category not in manifest or not manifest[category]:
        raise ValueError(f"no cached exemplars for {category!r} — run wikiart_cache first")

    entry = rng.choice(manifest[category])
    img = Image.open(entry["path"]).convert("RGB")
    detail = _crop_detail(img, rng)

    # Choices are only styles with cached exemplars — grows to 5 automatically once
    # wikiart_cache.py finishes the remaining categories, no code change needed.
    choices_keys = [s for s in _STYLE_INFO if manifest.get(s)]
    rng.shuffle(choices_keys)
    correct_idx = choices_keys.index(category)
    choices = [_STYLE_INFO[k]["display"] for k in choices_keys]

    seed = rng.randint(0, 10**6)
    return {
        "id": f"arh.stylesense.{seed:06d}",
        "course": "ARH",
        "category": "StyleSense",
        "subcategory": category,
        "stimulus": {"type": "pil_image", "image": detail},
        "prompt": "This is a detail from a painting. What style/period is it?",
        "choices": choices,
        "correct": correct_idx,
        "feedback": f"{_STYLE_INFO[category]['display']} — {_STYLE_INFO[category]['cue']} (artist: {entry['artist']})",
        "ground_truth_method": f"huggan/wikiart curated style label ({entry['path']})",
        "difficulty": difficulty,
        "transfer": False,
        "provenance": {"generator": "template_stylesense_v1", "seed": seed, "source_image": entry["path"]},
    }
