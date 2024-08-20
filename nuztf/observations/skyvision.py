"""
Skyvision coverage
"""

import json
from pathlib import Path

import pandas as pd
from astropy.time import Time
from tqdm import tqdm
from ztfquery import skyvision

from nuztf.observations.shared import coverage_dir, get_date, partial_flag


def coverage_skyvision_path(jd: float) -> Path:
    """
    Find the path to the Skyvision coverage file for a given JD

    :param jd: JD
    :return: Output path
    """
    if (Time.now().jd - jd) < 1:
        partial_ext = partial_flag
    else:
        partial_ext = ""

    output_path = coverage_dir.joinpath(f"{get_date(jd)}{partial_ext}_skyvision.json")

    return output_path


def write_coverage_skyvision(jds: list[float]):
    """
    Write the Skyvision coverage for a list of JDs to the cache

    :param jds: JDs
    :return: None
    """
    for jd in tqdm(jds):
        date = get_date(jd)

        assert jd - int(jd) == 0.5, "JD must be a half-integer"

        res = skyvision.get_log(f"{date[:4]}-{date[4:6]}-{date[6:8]}", verbose=False)

        path = coverage_skyvision_path(jd)

        if res is not None:
            # The skyvision log has a bug where some entries have a FieldID of "NONE"
            mask = (
                pd.notnull(res["FieldID"]) & res["FieldID"].astype(str).str.isnumeric()
            )
            res = res[mask]
            jds = [
                Time(f'{row["UT Date"]}T{row["UT Time"]}').jd
                for _, row in res.iterrows()
            ]
            res["obsjd"] = jds
            res["status"] = (res["Observation Status"] == "FAILED").astype(int)
            res["filter_id"] = res["Filter"].apply(
                lambda x: 1 if x == "FILTER_ZTF_G" else 2 if x == "FILTER_ZTF_R" else 3
            )
            res["maglim"] = 20.5
            res["field_id"] = res["FieldID"].astype(int)
            res["exposure_time"] = res["Exptime"]
            res = res[
                ["obsjd", "filter_id", "field_id", "exposure_time", "maglim", "status"]
            ]

            new_res = []
            for _, row in res.iterrows():
                new = row.to_dict()
                new_res += [dict(qid=int(i), **new) for i in range(64)]
            pd.DataFrame(new_res).to_json(path)

        else:
            with open(path, "w") as f:
                json.dump({}, f)
