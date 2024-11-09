#!/usr/bin/env python
# coding: utf-8

import json
import logging
import os
import time
from pathlib import Path

import healpy as hp
import lxml.etree
import numpy as np
import requests
import wget
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy_healpix import HEALPix
from ligo.gracedb.exceptions import HTTPError
from ligo.gracedb.rest import GraceDb
from ligo.skymap.io import read_sky_map
from ligo.skymap.moc import rasterize
from lxml import html
from numpy.lib.recfunctions import append_fields

from nuztf.paths import RESULTS_DIR, SKYMAP_DIR


class EventNotFound(Exception):
    """Base class for non-existing event"""


class RetractionError(Exception):
    """Base class for retracted event"""


class Skymap:
    def __init__(
        self,
        event: str = None,
        rev: int = None,
        prob_threshold: float = 0.9,
        output_nside: int | None = None,
    ):
        self.logger = logging.getLogger(__name__)

        self.prob_threshold = prob_threshold
        self.event = event

        if ".fit" in event:
            basename = Path(event).stem

            self.event_name = basename
            self.skymap_path = SKYMAP_DIR.joinpath(basename)
            self.rev = None

            if (event[:8] == "https://") | (event[:7] == "http://"):
                if not self.skymap_path.exists():
                    self.logger.info(f"Downloading from: {event}")
                    wget.download(event, str(self.skymap_path))
                else:
                    self.logger.info(f"Found saved skymap at: {self.skymap_path}")
        #
        # elif "grb" in event.lower():
        #     self.skymap_path, self.event_name = self.get_grb_skymap(event=event)

        else:
            self.skymap_path, self.event_name, self.rev = self.get_gw_skymap(
                event=event, rev=rev
            )

        (
            self.data,
            self.t_obs,
            self.hpm,
            self.key,
            self.dist,
            self.dist_unc,
        ) = self.read_map(output_nside=output_nside)

        t_min = Time(self.t_obs, format="isot", scale="utc")

        self.logger.info(f"Event time: {t_min}")
        self.logger.info("Reading map")

        self.pixel_threshold = self.find_pixel_threshold(self.data[self.key])

    # def get_grb_skymap(self, event: str):
    #     """ """
    #     if event is None:
    #         raise ValueError(
    #             "event_name must be provided for GRBs. They must have the form 'GRB210729A"
    #         )
    #
    #     event_year_short = event[3:5]
    #     event_year = "20" + event_year_short
    #     event_month = event[5:7]
    #     event_day = event[7:9]
    #     event_letter = event[9]
    #     event_number = ord(event_letter) - 65
    #
    #     # get possible skymap URLs
    #
    #     url = f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/{event_year}"
    #
    #     page_overview = requests.get(url)
    #     webpage_overview = html.fromstring(page_overview.content)
    #
    #     links_overview = webpage_overview.xpath("//a/@href")
    #
    #     links_for_date = []
    #
    #     for link in links_overview:
    #         if link[2:8] == f"{event_year_short}{event_month}{event_day}":
    #             links_for_date.append(url + "/" + link + "current/")
    #
    #     if len(links_for_date) > 1:
    #         self.logger.info(
    #             f"Found multiple events. "
    #             f"Will choose the one corresponding the GRB letter {event_letter}"
    #         )
    #
    #     event_url = links_for_date[event_number]
    #
    #     page_event = requests.get(event_url)
    #     webpage_event = html.fromstring(page_event.content)
    #     links_event = webpage_event.xpath("//a/@href")
    #
    #     for link in links_event:
    #         if link[0:11] == "glg_healpix":
    #             final_link = event_url + link
    #             break
    #
    #     self.skymap_path = os.path.join(self.base_skymap_dir, link)
    #
    #     if os.path.isfile(self.skymap_path):
    #         self.logger.info(
    #             f"Continuing with saved skymap. Located at {self.skymap_path}"
    #         )
    #     else:
    #         self.logger.info(f"Downloading skymap and saving to {self.skymap_path}")
    #         wget.download(final_link, self.skymap_path)
    #
    #     return sk, event

    def get_gw_skymap(self, event: str, rev: int | None = None):
        """
        Function to download a GW skymap from GraceDB

        :param event: Event name
        :param rev: revision number
        :return:
        """

        ligo_client = GraceDb()

        self.logger.info("Obtaining skymap from GraceDB")

        if event is None:
            superevent_iterator = ligo_client.superevents("category: Production")
            superevent_ids = [
                superevent["superevent_id"] for superevent in superevent_iterator
            ]
            event = superevent_ids[0]

        try:
            res = ligo_client.voevents(event)
            if res.status_code == 200:
                voevents = res.json()["voevents"]
            else:
                raise EventNotFound(
                    f"The specified LIGO event, {event}, was not found on GraceDB. "
                    f"Please check that you entered the correct event name."
                )
        except HTTPError:
            raise EventNotFound(
                f"The specified LIGO event, {event}, was not found on GraceDB. "
                f"Please check that you entered the correct event name."
            )

        if rev is None:
            rev = len(voevents)

        elif rev > len(voevents):
            raise Exception("Revision {0} not found".format(rev))

        latest_voevent = voevents[rev - 1]
        self.logger.info(f"Found voevent {latest_voevent['filename']}")

        if "Retraction" in latest_voevent["filename"]:
            raise RetractionError(
                f"The specified LIGO event, "
                f"{latest_voevent['filename']}, was retracted."
            )

        response = requests.get(latest_voevent["links"]["file"])

        root = lxml.etree.fromstring(response.content)
        params = {
            elem.attrib["name"]: elem.attrib["value"]
            for elem in root.iterfind(".//Param")
        }

        latest_skymap = params["skymap_fits"]

        self.logger.info(f"Latest skymap URL: {latest_skymap}")

        savepath = SKYMAP_DIR.joinpath(
            f"{event}_rev{latest_voevent['N']}_{os.path.basename(latest_skymap)}"
        )

        self.logger.info(f"Saving to: {savepath}")
        response = requests.get(latest_skymap)

        with open(savepath, "wb") as f:
            f.write(response.content)

        event_name = f"{event}/rev{latest_voevent['N']}"

        return savepath, event_name, rev

    def read_map(self, output_nside: int | None = None):
        """
        Read the skymap

        :param output_nside: The nside of the output skymap.
            If None, the nside of the input skymap will be used.

        """

        self.logger.info(f"Reading file: {self.skymap_path}")

        with fits.open(self.skymap_path) as hdul:
            data = None
            h = hdul[0].header

            dist = None
            dist_unc = None
            t_obs = None
            ordering = None

            for x in hdul:
                if data is None:
                    if x.data is not None:
                        data = np.array(x.data)

                if "DISTMEAN" in x.header:
                    dist = x.header["DISTMEAN"]

                if "DISTSTD" in x.header:
                    dist_unc = x.header["DISTSTD"]

                if "DATE-OBS" in x.header:
                    t_obs = x.header["DATE-OBS"]

                elif "EVENTMJD" in x.header:
                    t_obs_mjd = x.header["EVENTMJD"]
                    t_obs = Time(t_obs_mjd, format="mjd").isot

                if "ORDERING" in x.header:
                    ordering = x.header["ORDERING"]

        if "PROB" in data.dtype.names:
            pass
        elif (replacekey := "PROBABILITY") in data.dtype.names:
            prob = np.array(data[replacekey]).flatten()
            data = append_fields(data, "PROB", prob)
        elif (replacekey := "PROBDENSITY") in data.dtype.names:
            prob = np.array(data[replacekey])
            data = append_fields(data, "PROB", prob)
        elif (replacekey := "T") in data.dtype.names:  # weird IceCube format
            prob = np.array(data[replacekey]).flatten()
            data = append_fields(data, "PROB", prob)
        else:
            raise Exception(
                f"No recognised probability key in map. This is probably a weird one, right? "
                f"Found the following keys: {data.dtype.names}"
            )

        if ordering == "NUNIQ":
            self.logger.info("Rasterising skymap to convert to nested format")
            # We need to use the ligo.skymap.io map parser to make rasterize work
            skymap_uniq = read_sky_map(self.skymap_path, moc=True)
            if (replacekey := "PROBDENSITY") in skymap_uniq.colnames:
                skymap_uniq[replacekey].name = "PROB"

            data = rasterize(skymap_uniq, order=7)

        if not isinstance(data["PROB"][0], float):
            self.logger.info("Flattening skymap")
            prob = np.array(data["PROB"]).flatten()
            data = np.array(prob, dtype=np.dtype([("PROB", float)]))

        if "NSIDE" not in h.keys():
            h["NSIDE"] = hp.npix2nside(len(data["PROB"]))

        data["PROB"] /= np.sum(data["PROB"])

        self.logger.info(f"Summed probability is {100. * np.sum(data['PROB']):.1f}%")

        if ordering == "RING":
            data["PROB"] = hp.pixelfunc.reorder(data["PROB"], inp="RING", out="NESTED")

        if ordering is not None:
            h["ORDERING"] = "NESTED"

        else:
            raise Exception(
                f"Error parsing fits file, no ordering found. "
                f"Please enter the ordering (NESTED/RING/NUNIQ)"
            )

        # Optionally interpolate to a different nside
        if output_nside is not None:
            if output_nside != hp.npix2nside(len(data["PROB"])):
                self.logger.info(f"Regridding to nside {output_nside}")

                new_prob = hp.ud_grade(
                    data["PROB"],
                    nside_out=output_nside,
                    order_in=h["ORDERING"],
                    order_out="NESTED",
                    power=-2,
                )

                new_data = np.array(new_prob, dtype=np.dtype([("PROB", float)]))

                if "DISTMEAN" in h.keys():
                    dist = hp.ud_grade(
                        data["DISTMEAN"],
                        nside_out=output_nside,
                        order_in=h["ORDERING"],
                        order_out="NESTED",
                        power=0,
                    )

                    new_data["DISTMEAN"] = dist

                if "DISTSTD" in h.keys():
                    dist_unc = hp.ud_grade(
                        data["DISTSTD"],
                        nside_out=output_nside,
                        order_in=h["ORDERING"],
                        order_out="NESTED",
                        power=0,
                    )
                    new_data["DISTSTD"] = dist_unc

                data = new_data

        else:
            output_nside = h["NSIDE"]

        hpm = HEALPix(nside=output_nside, order="NESTED", frame="icrs")

        return data, t_obs, hpm, "PROB", dist, dist_unc

    def find_pixel_threshold(self, data):
        """
        Find the pixel threshold that corresponds to the probability threshold

        :param data: Healpix map data
        :return: threshold value
        """

        ranked_pixels = np.sort(data)[::-1]
        int_sum = 0.0
        pixel_threshold = 0.0

        for i, prob in enumerate(ranked_pixels):
            int_sum += prob
            if int_sum > self.prob_threshold:
                self.logger.info(
                    f"Threshold found! \n To reach {int_sum * 100.0:.2f}% of "
                    f"probability, pixels with probability greater "
                    f"than {prob:.3g} are included."
                )
                pixel_threshold = prob
                break

        return pixel_threshold

    def interpolate_map(self, ra_deg: float, dec_deg: float) -> float:
        """
        Find the probability at a given sky position by interpolating the skymap

        :param ra_deg: Right ascension in degrees
        :param dec_deg: declination in degrees
        :return: Value of the skymap at the given position
        """
        interpol_map = self.hpm.interpolate_bilinear_skycoord(
            SkyCoord(ra_deg * u.deg, dec_deg * u.deg), self.data[self.key]
        )
        return interpol_map

    def in_contour(self, ra_deg: float, dec_deg: float) -> bool:
        """
        Find if a given sky position is within the contour of the skymap

        :param ra_deg: Right ascension in degrees
        :param dec_deg: declination in degrees
        :return: bool
        """
        return self.interpolate_map(ra_deg, dec_deg) > self.pixel_threshold
