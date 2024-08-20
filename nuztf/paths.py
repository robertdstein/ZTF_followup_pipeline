import os
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.joinpath("config")

NUZTF_OUTPUT_DIR_STR = os.environ.get("NUZTF_DIR")

if NUZTF_OUTPUT_DIR_STR is None:
    NUZTF_OUTPUT_DIR = Path.home() / "Data" / "nuztf"
else:
    NUZTF_OUTPUT_DIR = Path(NUZTF_OUTPUT_DIR_STR)

SKYMAP_DIR = NUZTF_OUTPUT_DIR / "skymaps"
SKYMAP_DIR.mkdir(exist_ok=True, parents=True)

RESULTS_DIR = NUZTF_OUTPUT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR = NUZTF_OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
BASE_CANDIDATE_DIR = CACHE_DIR / "candidates"
BASE_CANDIDATE_DIR.mkdir(exist_ok=True, parents=True)

CROSSMATCH_CACHE = CACHE_DIR / "crossmatch"
CROSSMATCH_CACHE.mkdir(exist_ok=True, parents=True)

CUTOUT_CACHE_DIR = CACHE_DIR / "cutouts"
CUTOUT_CACHE_DIR.mkdir(exist_ok=True, parents=True)

FLATPIX_CACHE_DIR = CACHE_DIR / "flatpix"
FLATPIX_CACHE_DIR.mkdir(exist_ok=True, parents=True)

PREPROCESSED_CACHE_DIR = CACHE_DIR / "preprocessed"
PREPROCESSED_CACHE_DIR.mkdir(exist_ok=True, parents=True)
