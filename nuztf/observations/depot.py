import json
import logging
from pathlib import Path

import requests
from astropy.time import Time
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

from nuztf import credentials
from nuztf.observations.shared import NoDepotEntry, coverage_dir, get_date, partial_flag

logger = logging.getLogger(__name__)

username, password = credentials.load_credentials("ipacdepot")
data = {"username": username, "password": password}


def download_depot_log(date):
    """
    Download the depot log for a given date

    :param date: date in YYYYMMDD format
    :return: json log
    """
    url = f"https://ztfweb.ipac.caltech.edu/ztf/depot/{date}/ztf_recentproc_{date}.json"
    response = requests.get(url, auth=HTTPBasicAuth(username, password))
    if response.status_code == 404:
        raise NoDepotEntry(f"No depot entry for {date} at url {url}")
    response.raise_for_status()
    return response.json()


def coverage_depot_path(jd: float) -> Path:
    """
    Return the path to the cached coverage file for a given JD

    :param jd: JD
    :return: path to cached coverage file
    """

    date = get_date(jd)

    if (Time.now().jd - jd) < 1:
        partial_ext = partial_flag
    else:
        partial_ext = ""

    return coverage_dir.joinpath(f"{date}{partial_ext}.json")


def write_coverage_depot(jds: list[float]):
    """
    Write the depot coverage for a list of JDs to the cache

    Requires a valid IPAC depot login

    :param jds: JDs
    :return: None
    """
    for jd in tqdm(jds):
        date = get_date(jd)
        try:
            log = download_depot_log(date)
        except NoDepotEntry as exc:
            log = {}
            logger.warning(f"No depot entry for {date}: {exc}")

        path = coverage_depot_path(jd)
        with open(path, "w") as f:
            json.dump(log, f)
