import logging
import time
from pathlib import Path

import backoff
import numpy as np
import pyvo.dal
import requests
from astropy.time import Time
from pyvo.auth import authsession, securitymethods
from pyvo.utils.http import create_session

from nuztf.credentials import load_credentials
from nuztf.observations.shared import coverage_dir, get_date, partial_flag

logger = logging.getLogger(__name__)


# IPAC TAP login
username, password = load_credentials("irsa")
data = {"username": username, "password": password}
headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
IPAC_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"

# The following functions are used to find the paths to the cached coverage files


def coverage_tap_path(jd: float) -> Path:
    """
    Find the path to the TAP coverage file for a given JD

    :param jd: JD
    :return: Output path
    """
    if (Time.now().jd - jd) < 1:
        partial_ext = partial_flag
    else:
        partial_ext = ""

    output_path = coverage_dir.joinpath(f"{get_date(jd)}{partial_ext}_tap.json")

    return output_path


# The following functions are used to write the coverage to the cache


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_time=60)
def write_coverage_tap(jds: [float]):
    """
    Write the coverage for a list of JDs to the cache using TAP

    (JDs must be half-integer)

    :param jds: Half-integer JDs
    :return: None
    """
    with create_session() as session:
        session.auth = (username, password)

        # Setup AuthSession
        auth = authsession.AuthSession()
        auth.credentials.set(securitymethods.BASIC, session)
        auth.add_security_method_for_url(IPAC_TAP_URL, securitymethods.BASIC)
        auth.add_security_method_for_url(IPAC_TAP_URL + "/sync", securitymethods.BASIC)
        auth.add_security_method_for_url(IPAC_TAP_URL + "/async", securitymethods.BASIC)
        auth.add_security_method_for_url(
            IPAC_TAP_URL + "/tables", securitymethods.BASIC
        )

        # Create service
        client = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP", session)

        for jd in jds:
            assert jd - int(jd) == 0.5, "JD must be a half-integer"

            obstable = client.search(
                f"""
            SELECT expid,obsjd,fid,field,exptime,rcid,maglimit,infobits,ipac_gid
            FROM ztf.ztf_current_meta_sci WHERE (obsjd BETWEEN {jd} AND {jd + 1})
            """
            ).to_table()
            names = (
                "expid",
                "exptime",
                "field",
                "fid",
                "rcid",
                "maglimit",
                "infobits",
                "ipac_gid",
            )
            renames = (
                "exposure_id",
                "exposure_time",
                "field_id",
                "filter_id",
                "qid",
                "maglim",
                "status",
                "programid",
            )
            obstable.rename_columns(names, renames)

            obs = obstable.to_pandas()
            obs["nalertpackets"] = np.nan

            output_path = coverage_tap_path(jd)

            obs.to_json(output_path)
            time.sleep(1)

            logger.info(f"Coverage for JD {jd} written to {output_path} using TAP.")
