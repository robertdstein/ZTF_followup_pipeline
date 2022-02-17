#!/usr/bin/env python
# coding: utf-8

import unittest
import logging
from astropy.time import Time
from astropy import units as u
from nuztf.observations import get_obs_summary, get_obs_summary_2


class TestCoverage(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def test_lightcurve(self):
        self.logger.info("\n\n Testing observation log parsing \n\n")

        res = get_obs_summary(
            Time(2458865.96, format="jd"),
            Time(2458867.96, format="jd")
        )

        expected = {
            "obsid": 111223429.0,
            "field": 3.550000e+02,
            "obsjd": 2458866.734294,
            "seeing": 3.4250149727,
            "limmag": 19.998298645,
            "exposure_time": 3.000000e+01,
            "fid": 2.000000e+00,
            "processed_fraction": 1.000000e+00
        }

        self.assertEqual(len(res.data), 1127)

        for (name, val) in expected.items():
            self.assertEqual(res.data.iloc[0][name], val)

        # res2 = get_obs_summary_2(
        #     Time(2458865.96, format="jd"),
        #     Time(2458867.96, format="jd")
        # )
        #
        # print(res2)

        get_obs_summary(
            Time.now()-5.*u.day,
            Time.now()
        )




