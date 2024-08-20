import logging

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from tqdm import tqdm

from nuztf.observations.depot import coverage_depot_path, write_coverage_depot
from nuztf.observations.mns import MNS
from nuztf.observations.shared import coverage_dir, partial_flag
from nuztf.observations.skyvision import (
    coverage_skyvision_path,
    write_coverage_skyvision,
)
from nuztf.observations.tap import coverage_tap_path, write_coverage_tap

logger = logging.getLogger(__name__)


# The following functions are used to get the coverage from the cache


def get_coverage(jds: [int], backend="best") -> pd.DataFrame | None:
    """
    Get a dataframe of the coverage for a list of JDs

    Will use the cache if available, otherwise will query the depot, and lastly TAP

    :param jds: JDs
    :param backend: "best" or "depot" or "tap" or "skyvision" or "masterlog"
    :return: Coverage dataframe
    """

    assert backend in [
        "best",
        "depot",
        "tap",
        "skyvision",
        "masterlog",
    ], f"Invalid backend '{backend}'"

    # Clear any logs flagged as partial/incomplete

    cache_files = coverage_dir.glob("*.json")
    partial_logs = [x for x in cache_files if partial_flag in str(x)]

    if len(partial_logs) > 0:
        logger.debug(f"Removing the following partial logs: {partial_logs}")
        for partial_log in partial_logs:
            partial_log.unlink()

    # Only write missing logs

    covered_jds = []

    if backend in ["best", "depot"]:
        for jd in jds:
            if jd not in covered_jds:
                depot_path = coverage_depot_path(jd)
                if depot_path.exists():
                    df = pd.read_json(depot_path)
                    if len(df) > 0:
                        covered_jds.append(jd)

        missing_logs = sorted(set(jds) - set(covered_jds))

        if len(missing_logs) > 0:
            logger.info(
                f"Some logs were missing from the cache. "
                f"Querying for the following JDs in depot: {missing_logs}"
            )
            write_coverage_depot(missing_logs)

    if backend in ["best", "skyvision"]:
        for jd in jds:
            if jd not in covered_jds:
                skyvision_path = coverage_skyvision_path(jd)
                if skyvision_path.exists():
                    df = pd.read_json(coverage_skyvision_path(jd))
                    if len(df) > 0:
                        covered_jds.append(jd)

        missing_logs = sorted(set(jds) - set(covered_jds))

        if len(missing_logs) > 0:
            logger.info(
                f"Some logs were still missing from the cache. "
                f"Querying for the following JDs from skyvision: {missing_logs}"
            )
            write_coverage_skyvision(missing_logs)

    # Try TAP for missing logs

    if backend in ["best", "tap"]:
        for jd in jds:
            if jd not in covered_jds:
                tap_path = coverage_tap_path(jd)
                if tap_path.exists():
                    df = pd.read_json(tap_path)
                    if len(df) > 0:
                        covered_jds.append(jd)

        missing_logs = sorted(set(jds) - set(covered_jds))

        if len(missing_logs) > 0:
            logger.info(
                f"Some logs were still missing from the cache. "
                f"Querying for the following JDs from TAP: {missing_logs}"
            )
            write_coverage_tap(missing_logs)

    # Load logs from cache

    results = []

    for jd in tqdm(jds):

        res = None

        if backend in ["best", "depot"]:
            df = pd.read_json(coverage_depot_path(jd))
            if len(df) > 0:
                res = df

        if backend in ["best", "skyvision"]:
            if res is None:
                df = pd.read_json(coverage_skyvision_path(jd))
                if len(df) > 0:
                    res = df

        if backend in ["best", "tap"]:
            if res is None:
                df = pd.read_json(coverage_tap_path(jd))
                if len(df) > 0:
                    res = df

        if res is not None:
            results.append(res)

    if results:
        result_df = pd.concat(results, ignore_index=True)
        return result_df
    else:
        return None


def get_obs_summary_depot(t_min: Time, t_max: Time, backend="best") -> MNS | None:
    """
    Get observation summary from depot

    :param t_min: Start time
    :param t_max: End time
    :param backend: "best" or "depot" or "tap" or "skyvision"
    :return: MNS object
    """

    jds = np.arange(int(t_min.jd) - 0.5, int(t_max.jd) + 1.5)

    res = get_coverage(jds, backend=backend)

    if len(res) == 0:
        return None

    res["date"] = Time(res["obsjd"].to_numpy(), format="jd").isot

    mns = MNS(df=res)

    mns.data.query(f"obsjd >= {t_min.jd} and obsjd <= {t_max.jd}", inplace=True)

    mns.data.reset_index(inplace=True)
    mns.data.drop(columns=["index"], inplace=True)

    return mns


def get_obs_summary(
    t_min,
    t_max=None,
    max_days: float = None,
    backend="best",
) -> MNS | None:
    """
    Get observation summary from IPAC depot

    :param t_min: Start time
    :param t_max: End time
    :param max_days: Maximum number of days
    :param backend: "best" or "depot" or "tap" or "skyvision"
    :return: MNS object
    """
    now = Time.now()

    if t_max and max_days:
        raise ValueError("Choose either t_max or max_days, not both")

    if t_max is None:
        if max_days is None:
            t_max = now
        else:
            t_max = t_min + (max_days * u.day)

    if t_max > now:
        t_max = now

    logger.info(f"Getting observation logs  using backend {backend}.")
    mns = get_obs_summary_depot(t_min=t_min, t_max=t_max, backend=backend)

    if mns is not None:
        logger.info(f"Found {len(set(mns.data))} observations in total.")
    else:
        logger.warning(f"Found no observations using backend {backend}.")

    return mns
