"""
Module for handling the master log of observations.
"""

import pandas as pd
import requests
from astropy.time import Time
from requests.auth import HTTPBasicAuth

from nuztf.credentials import load_credentials
from nuztf.paths import ZTF_LOG_PATH

MASTER_LOG_URL = "https://sites.astro.caltech.edu/~tb/ztfops/sky/allexp.tbl"

username, password = load_credentials("ztfops")


def download_master_log():
    """
    Download the master log of observations
    """
    response = requests.get(MASTER_LOG_URL, auth=HTTPBasicAuth(username, password))
    response.raise_for_status()
    with open(ZTF_LOG_PATH, "wb") as f:
        f.write(response.content)


def load_master_log() -> pd.DataFrame:
    """
    Load the master log of observations
    """
    df = pd.read_fwf(ZTF_LOG_PATH, comment="|")
    mask = df["type"] == "targ"
    df = df[mask]
    mask = df["p"].astype(float) > 0
    df = df[mask]
    df.rename(
        columns={"l": "filter_id", "ld": "field_id", "p": "exposure_time", "id": "pid"},
        inplace=True,
    )
    obs = Time(df["UT_START"].to_list(), format="isot")
    df["obsjd"] = obs.jd
    df["status"] = 0
    df["maglim"] = 20.5

    df = df[["obsjd", "filter_id", "field_id", "exposure_time", "maglim", "status"]]
    df.reset_index(inplace=True, drop=True)
    return df
