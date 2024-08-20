#!/usr/bin/env python3
# coding: utf-8

import logging

import healpy as hp
import numpy as np
import yaml
from astropy.time import Time
from tqdm import tqdm

from nuztf.base_scanner import BaseScanner
from nuztf.parse_nu_gcn import find_gcn_no, get_latest_gcn, parse_gcn_circular
from nuztf.paths import CONFIG_DIR


class NeutrinoScanner(BaseScanner):
    default_fritz_group = 221

    def __init__(
        self,
        nu_name: str = None,
        manual_args=None,
        gcn_no: int = None,
        cone_nside: int = 128,
        t_precursor: float = None,
        config: dict = None,
        output_nside: int = 1024,
    ):
        self.logger = logging.getLogger(__name__)

        if config:
            self.config = config
        else:
            config_path = CONFIG_DIR.joinpath("nu_run_config.yaml")
            with open(config_path) as f:
                self.config = yaml.safe_load(f)

        self.prob_threshold = 0.9

        if manual_args is None:
            if nu_name is not None:
                gcn_no = find_gcn_no(nu_name)

            elif gcn_no is None:
                gcn_no = get_latest_gcn()

            gcn_info = parse_gcn_circular(gcn_no)

            self.logger.info(gcn_info)

            nu_name_gcn = gcn_info.get("name")
            if nu_name_gcn is None:
                raise ValueError(f"No GCN notice or circular found for {nu_name}.")
            else:
                nu_name = nu_name_gcn
            author = gcn_info["author"]
            ra = [gcn_info["ra"], gcn_info["ra_err"][0], gcn_info["ra_err"][1]]
            dec = [gcn_info["dec"], gcn_info["dec_err"][0], gcn_info["dec_err"][1]]
            nu_time = gcn_info["time"]

            if t_precursor is not None:
                nu_time = Time(nu_time.mjd - t_precursor, format="mjd")

        else:
            (nu_name, ra, dec, nu_time) = manual_args
            author = None

        self.nu_name = nu_name
        self.author = author
        self.gcn_no = gcn_no
        self.dist = None

        self.logger.info(f"Neutrino time: {nu_time}")

        self.ra_max = float(max(ra[1:]) + ra[0])
        self.ra_min = float(min(ra[1:]) + ra[0])
        self.dec_max = float(max(dec[1:]) + dec[0])
        self.dec_min = float(min(dec[1:]) + dec[0])

        self.logger.info(f"Coordinates: RA = {ra[0]} ({self.ra_min} - {self.ra_max})")
        self.logger.info(
            f"Coordinates: Dec = {dec[0]} ({self.dec_min} - {self.dec_max})"
        )

        BaseScanner.__init__(
            self,
            t_min=nu_time,
            run_config=self.config,
            cone_nside=cone_nside,
            output_nside=output_nside,
        )
        self.prob_threshold = 0.9
        self.rectangular_area = (
            (self.ra_max - self.ra_min)
            * (self.dec_max - self.dec_min)
            * abs(np.cos(np.radians(dec[0])))
        )

        self.logger.info(f"Projected Area: {self.rectangular_area:.3f} sq. deg.")

    def get_name(self):
        """ """
        return self.nu_name

    def get_full_name(self):
        """ """
        return (
            f"neutrino event {self.get_name()} "
            f"({self.author} et. al, GCN {self.gcn_no})"
        )

    @staticmethod
    def get_obs_line():
        """ """
        return "Each exposure was 300s with a typical depth of 21.0 mag."

    @staticmethod
    def remove_variability_line():
        """ """
        return ""

    def filter_f_no_prv(self, res: dict):
        """ """

        ztf_id = res["objectId"]

        # Positive detection
        if res["candidate"]["isdiffpos"] not in ["t", "1"]:
            self.logger.debug(f"❌ {ztf_id}: Negative subtraction")
            return False

        try:
            if res["candidate"]["drb"] < 0.3:
                self.logger.debug(f"❌ {ztf_id}: DRB too low")
                return False
        except (KeyError, TypeError) as e:
            pass

        # Check contour
        if not self.in_contour(res["candidate"]["ra"], res["candidate"]["dec"]):
            self.logger.debug(f"❌ {ztf_id}: Not in contour")
            return False

        endhist = res["candidate"]["jdendhist"]
        starthist = res["candidate"]["jdstarthist"]

        if endhist == starthist:
            self.logger.debug(f"❌ {ztf_id}: One detection only")
            return False

        # Require 2 detections separated by 15 mins
        if (endhist - starthist) < 0.01:
            self.logger.debug(
                f"❌ {ztf_id}: Does have 2 detections, but these are not separated by "
                f">15 mins (delta t = {(endhist-starthist)*1440:.0f} min)"
            )
            return False

        self.logger.debug(f"✅ {ztf_id}: Passes first filtering stage (no prv).")
        return True

    def filter_f_history(self, res: dict):
        """
        Filter based on 2 detection requirement
        and probability contour requirement
        """

        ztf_id = res["objectId"]

        n_detections = len(
            [x for x in res["prv_candidates"] if "isdiffpos" in x.keys()]
        )

        if n_detections < 1:
            self.logger.info(f"❌ {ztf_id}: Has insufficient detections")
            return False

        if not self.in_contour(res["candidate"]["ra"], res["candidate"]["dec"]):
            self.logger.info(f"❌ {ztf_id}: Not in contour")
            return False

        self.logger.debug(f"✅ {ztf_id}: Passed the history filtering stage")

        return True

    def find_cone_coords(self):
        """ """
        cone_coords = []
        cone_ids = []

        scan_radius = np.degrees(hp.max_pixrad(self.cone_nside))

        self.logger.info("Finding search pixels:")

        for i in tqdm(range(hp.nside2npix(self.cone_nside))):
            ra, dec = self.extract_ra_dec(self.cone_nside, i)
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            if np.logical_and(
                ra > self.ra_min - scan_radius, ra < self.ra_max + scan_radius
            ):
                if np.logical_and(
                    dec > self.dec_min - scan_radius,
                    dec < self.dec_max + scan_radius,
                ):
                    cone_coords.append((ra_rad, dec_rad))
                    cone_ids.append(i)

        cone_coords = np.array(
            cone_coords, dtype=np.dtype([("ra", float), ("dec", float)])
        )

        return cone_ids, cone_coords

    def in_contour(self, ra_deg, dec_deg):
        in_ra = np.logical_and(ra_deg > self.ra_min, ra_deg < self.ra_max)
        in_dec = np.logical_and(dec_deg > self.dec_min, dec_deg < self.dec_max)

        return np.logical_and(in_ra, in_dec)

    def unpack_skymap(self, output_nside: None | int = None):
        """
        Unpack the skymap and return the pixel coordinates and probabilities

        :param output_nside: Nside of the output map
        :return: Map coordinates, pixel numbers, output nside,
        map probabilities, data, total pixel area, key
        """
        output_nside = 2048 if output_nside is None else output_nside

        map_coords = []
        pixel_nos = []

        center_ra = np.radians(np.mean([self.ra_max, self.ra_min]))
        center_dec = np.radians(np.mean([self.dec_max, self.dec_min]))
        # Take the larger of the two sides and convert to radians
        # To make sure to include all pixels until the edge of the rectangle,
        # we have to devide by sqrt(2)
        # (not 2 as previously done here!)
        rad = np.radians(
            max(self.ra_max - self.ra_min, self.dec_max - self.dec_min)
        ) / np.sqrt(2)

        nearish_pixels = list(
            hp.query_disc(
                nside=output_nside,
                vec=hp.ang2vec(np.pi / 2.0 - center_dec, center_ra),
                radius=rad,
                nest=True,
            )
        )

        for i in tqdm(nearish_pixels):
            ra, dec = self.extract_ra_dec(output_nside, i)
            if self.in_contour(ra, dec):
                map_coords.append((ra, dec))
                pixel_nos.append(i)

        map_probs = np.ones_like(pixel_nos, dtype=float)
        map_probs /= np.sum(map_probs)

        key = "PROB"

        data = np.zeros(hp.nside2npix(output_nside), dtype=np.dtype([(key, float)]))
        data[np.array(pixel_nos)] = map_probs

        total_pixel_area = hp.nside2pixarea(output_nside, degrees=True) * float(
            len(map_coords)
        )

        return (
            map_coords,
            pixel_nos,
            output_nside,
            map_probs,
            data,
            total_pixel_area,
            key,
        )
