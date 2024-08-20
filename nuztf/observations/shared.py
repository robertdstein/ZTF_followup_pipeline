"""
Shared variables for the observations module
"""

from pathlib import Path

from astropy.time import Time
from ztfquery.io import LOCALSOURCE

coverage_dir = Path(LOCALSOURCE).joinpath("all_obs")
coverage_dir.mkdir(exist_ok=True, parents=True)

partial_flag = "_PARTIAL"


class NoDepotEntry(Exception):
    """
    No entry in the depot for a given date
    """


def get_date(jd: float) -> str:
    """
    Get the date in YYYYMMDD format from a JD

    :param jd: JD
    :return: String date
    """
    return str(Time(jd, format="jd").isot).split("T")[0].replace("-", "")
