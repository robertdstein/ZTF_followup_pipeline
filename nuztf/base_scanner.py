#!/usr/bin/env python3
# coding: utf-8

import json
import logging
import time
from pathlib import Path

import backoff
import healpy as hp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from ampel.ztf.alert.ZiAlertSupplier import ZiAlertSupplier
from ampel.ztf.dev.DevAlertConsumer import DevAlertConsumer
from ampel.ztf.t0.DecentFilter import DecentFilter
from astropy import units as u
from astropy.coordinates import Distance, SkyCoord
from astropy.time import Time
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from ztfquery import fields as ztfquery_fields

from nuztf.ampel_api import (
    ampel_api_acknowledge_chunk,
    ampel_api_lightcurve,
    ampel_api_name,
    ampel_api_skymap,
)
from nuztf.cat_match import ampel_api_tns, get_cross_match_info, query_ned_for_z
from nuztf.flatpix import get_flatpix, get_nested_pix
from nuztf.fritz import save_source_to_group
from nuztf.observations import get_obs_summary
from nuztf.paths import BASE_CANDIDATE_DIR, RESULTS_DIR
from nuztf.plot import lightcurve_from_alert
from nuztf.utils import cosmo

DEBUG = False
RATELIMIT_CALLS = 10
RATELIMIT_PERIOD = 1


class BaseScanner:
    default_fritz_group = 1430

    def __init__(
        self,
        run_config,
        t_min,
        resource=None,
        filter_class=DecentFilter,
        cone_nside=64,
        cones_to_scan=None,
        logger=None,
    ):
        self.cone_nside = cone_nside
        self.t_min = t_min
        (
            self.map_coords,
            self.pixel_nos,
            self.nside,
            self.map_probs,
            self.data,
            self.total_pixel_area,
            self.key,
        ) = self.unpack_skymap()

        if not hasattr(self, "prob_threshold"):
            self.prob_threshold = None

        if resource is None:
            resource = {
                "ampel-ztf/catalogmatch": "https://ampel.zeuthen.desy.de/api/catalogmatch/",
            }

        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.logger.info("AMPEL run config:")
        self.logger.info(run_config)

        lvl = self.logger.level

        if lvl > 10:
            logger_ampel = logging.getLogger("AMPEL_filter")
            logger_ampel.setLevel(logging.WARNING)
        else:
            from ampel.log.AmpelLogger import AmpelLogger

            logger_ampel = AmpelLogger()

        self.ampel_filter_class = filter_class(
            logger=logger_ampel, resource=resource, **run_config
        )
        self.ampel_filter_class.post_init()

        self.dac = DevAlertConsumer(self.ampel_filter_class)

        self.scanned_pixels = []

        if cones_to_scan is None:
            self.cone_ids, self.cone_coords = self.find_cone_coords()
        else:
            self.cone_ids, self.cone_coords = cones_to_scan

        self.cache = dict()
        self.default_t_max = t_min + 10.0

        self.overlap_prob = None
        self.overlap_fields = None
        self.first_obs = None
        self.last_obs = None
        self.n_fields = None
        self.rectangular_area = None
        self.double_extragalactic_area = None

        self.observations = None

        self.pixel_area = hp.pixelfunc.nside2pixarea(self.nside, degrees=True)

        if not hasattr(self, "dist"):
            self.dist = None

    def get_full_name(self) -> str:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError

    def get_output_dir(self) -> Path:
        output_dir = RESULTS_DIR.joinpath(self.get_name())
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    def get_cache_dir(self) -> Path:
        cache_dir = BASE_CANDIDATE_DIR.joinpath(self.get_name().replace(" ", "_"))
        cache_dir.mkdir(exist_ok=True, parents=True)
        return cache_dir

    def unpack_skymap(self):
        raise NotImplementedError

    @staticmethod
    def get_tiling_line():
        return ""

    @staticmethod
    def get_obs_line():
        raise NotImplementedError

    @staticmethod
    def remove_variability_line():
        raise NotImplementedError

    @staticmethod
    def fid_to_band(fid: int):
        fid_band = {1: "g", 2: "r", 3: "i"}
        return fid_band[fid]

    @staticmethod
    def parse_ztf_filter(fid: int):
        """
        Convert ZTF filter id to string

        :param fid: filter id
        :return: g, r, or i
        """
        return ["g", "r", "i"][fid - 1]

    def get_overlap_line(self):
        """ """
        if (self.overlap_prob is not None) and (
            self.double_extragalactic_area is not None
        ):
            return (
                f"We covered {self.overlap_prob:.1f}% "
                f"({self.double_extragalactic_area:.1f} sq deg) "
                f"of the reported localization region. "
                "This estimate accounts for chip gaps. "
            )
        else:
            self.logger.warning("No overlap line added!")
            return ""

    def filter_ampel(self, res):
        self.logger.debug(f"{res['objectId']}: Running AMPEL filter")

        shaped_alert = ZiAlertSupplier.shape_alert_dict(res, ["FilterTest"])
        filterres = self.ampel_filter_class.process(alert=shaped_alert)
        if filterres:
            return True
        else:
            return False

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_time=600,
    )
    def add_res_to_cache(self, res):
        for res_alert in res:
            if res_alert["objectId"] not in self.cache.keys():
                self.cache[res_alert["objectId"]] = res_alert
            elif (
                res_alert["candidate"]["jd"]
                > self.cache[res_alert["objectId"]]["candidate"]["jd"]
            ):
                self.cache[res_alert["objectId"]] = res_alert

    def add_to_cache_by_names(self, ztf_ids: list):
        for ztf_id in ztf_ids:
            query_res = ampel_api_name(ztf_id, logger=self.logger)
            self.add_res_to_cache(query_res)

    def check_ampel_filter(self, ztf_name):
        lvl = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.DEBUG)
        self.logger.info("Set logger level to DEBUG")
        all_query_res = ampel_api_name(ztf_name, logger=self.logger)
        assert len(all_query_res) > 0, f"No results from ampel api for {ztf_name}"
        pipeline_bool = False
        for query_res in all_query_res:
            self.logger.info("Checking filter f (no prv)")
            no_prv_bool = self.filter_f_no_prv(query_res)
            self.logger.info(f"Filter f (np prv): {no_prv_bool}")
            if no_prv_bool:
                self.logger.info("Checking ampel filter")
                bool_ampel = self.filter_ampel(query_res)
                self.logger.info(f"ampel filter: {bool_ampel}")
                if bool_ampel:
                    self.logger.info("Checking filter f (history)")
                    history_bool = self.filter_f_history(query_res)
                    self.logger.info(f"Filter f (history): {history_bool}")
                    if history_bool:
                        pipeline_bool = True
        self.logger.info(f"Setting logger back to {lvl}")
        logging.getLogger().setLevel(lvl)
        return pipeline_bool

    def get_multi_night_summary(self, max_days=None):
        mns = get_obs_summary(self.t_min, max_days=max_days)
        # if mns is None:
        #     mns = get_obs_summary(self.t_min, max_days=max_days)
        return mns

    def query_ampel(
        self,
        t_min=None,
        t_max=None,
    ):
        if t_max is None:
            t_max = self.default_t_max

        if t_min is None:
            t_min = self.t_min

        self.logger.info("Commencing skymap scan")

        self.logger.debug(
            f"API skymap search: nside = {self.cone_nside} / "
            f"# pixels = {len(self.cone_ids)} / "
            f"timespan = {t_max.jd-t_min.jd:.1f} days."
        )

        query_res = []

        resume = True
        chunk_size = 2000
        resume_token = None

        i = 0
        total_chunks = 0
        t0 = time.time()

        while resume:
            res, resume_token, chunk_id, remaining_chunks = ampel_api_skymap(
                pixels=self.cone_ids,
                nside=self.cone_nside,
                t_min_jd=t_min.jd,
                t_max_jd=t_max.jd,
                logger=self.logger,
                chunk_size=chunk_size,
                resume_token=resume_token,
                warn_exceeding_chunk=False,
            )
            query_res.extend(res)

            ampel_api_acknowledge_chunk(resume_token=resume_token, chunk_id=chunk_id)

            if i == 0:
                total_chunks = remaining_chunks + 1
                self.logger.info(f"Total chunks: {total_chunks}")

            if remaining_chunks % 50 == 0 and remaining_chunks != 0:
                t1 = time.time()
                processed_chunks = total_chunks - remaining_chunks
                time_per_chunk = (t1 - t0) / processed_chunks
                remaining_time = time_per_chunk * remaining_chunks
                self.logger.info(
                    f"Remaining chunks: {remaining_chunks}. Estimated time to finish: "
                    f"{remaining_time/60:.0f} min"
                )

            if len(res) < chunk_size:
                resume = False
                self.logger.info("Done.")
            else:
                self.logger.debug(
                    f"Chunk size reached ({chunk_size}), commencing next query."
                )
            i += 1

        self.logger.info(f"Ingested {len(query_res)} alerts. Commencing filtering now.")

        return query_res

    def get_initial_cache_path(self) -> Path:
        """
        Get the path to the initial cache file

        :return: Cache file path
        """
        return self.get_cache_dir().joinpath("initial_stage.json")

    def get_final_cache_path(self) -> Path:
        """
        Get the path to the final cache file

        :return: Cache file path
        """
        return self.get_cache_dir().joinpath("final_stage.json")

    def load_from_cache(self):
        """
        Load the cache from the final cache file

        :return: None
        """
        if not self.get_final_cache_path().exists():
            err = (
                f"Final cache file {self.get_final_cache_path()} does not exist. "
                f"Run scan_area first."
            )
            self.logger.error(err)
            raise FileNotFoundError(err)

        with open(self.get_final_cache_path(), "r") as infile:
            results = json.load(infile)

        self.add_results(results)

    def scan_area(
        self,
        t_min=None,
        t_max=None,
    ):
        """
        Retrieve alerts for the healpix map from AMPEL API,
        filter the candidates and create a summary
        """
        query_res = self.query_ampel(t_min=t_min, t_max=t_max)

        with open(self.get_initial_cache_path(), "w") as outfile:
            json.dump(query_res, outfile)

        ztf_ids_first_stage = []
        for res in tqdm(query_res):
            if self.filter_f_no_prv(res):
                if self.filter_ampel(res):
                    ztf_ids_first_stage.append(res["objectId"])

        ztf_ids_first_stage = list(set(ztf_ids_first_stage))

        self.logger.info(
            f"{len(ztf_ids_first_stage)} candidates survive filtering stage 1"
        )

        self.logger.info(f"Retrieving alert history from AMPEL for filtering stage 2")

        results = self.ampel_object_search(ztf_ids=ztf_ids_first_stage)

        with open(self.get_final_cache_path(), "w") as outfile:
            json.dump(results, outfile)

        self.add_results(results)
        self.create_candidate_summary()

    def add_results(self, results):
        """
        Add the results to the cache and create a summary

        :param results: Results from the AMPEL API
        :return: None
        """
        for res in results:
            self.add_res_to_cache(res)

        self.logger.info(f"Found {len(self.cache)} candidates")

    def filter_f_no_prv(self, res):
        raise NotImplementedError

    def filter_f_history(self, res):
        raise NotImplementedError

    def find_cone_coords(self):
        raise NotImplementedError

    @staticmethod
    def wrap_around_180(ra_deg: float):
        """ """
        ra_rad = np.deg2rad(ra_deg)
        ra_rad[ra_rad > np.pi] -= 2 * np.pi
        ra_deg = np.rad2deg(ra_rad)
        return ra_deg

    def in_contour(self, ra, dec):
        raise NotImplementedError

    def ampel_object_search(self, ztf_ids: list) -> list:
        """ """
        all_results = []

        for ztf_id in tqdm(ztf_ids):
            # get the full lightcurve from the API
            query_res = ampel_api_lightcurve(ztf_name=ztf_id, logger=self.logger)

            final_res = []

            for res in query_res:
                if self.filter_f_history(res):
                    final_res.append(res)

            all_results.append(final_res)

        return all_results

    @staticmethod
    def calculate_abs_mag(mag: float, redshift: float) -> float:
        """ """
        luminosity_distance = cosmo.luminosity_distance(redshift).value * 10**6
        abs_mag = mag - 5 * (np.log10(luminosity_distance) - 1)
        return abs_mag

    def get_candidates_lines(self) -> str:
        """
        Return a string with the candidates in a format that can be
        directly pasted into a Slack message or GCN

        Returns: str
        """

        if len(self.cache) > 0:
            s = (
                "We are left with the following high-significance transient "
                "candidates by our pipeline, all lying within the "
                f"{100 * self.prob_threshold}% localization of the skymap."
                f"\n\n{self.parse_candidates()}"
            )
        else:
            s = "\n\nNo candidate counterparts were detected."

        return s

    def parse_candidates(self):
        """
        Parse the candidates into a string that can be pasted into a Slack message
        or GCN

        :return: Str
        """

        table = (
            "+--------------------------------------------------------------------------------+\n"
            "| ZTF Name     | IAU Name  | RA (deg)    | DEC (deg)   | Filter | Mag   | MagErr |\n"
            "+--------------------------------------------------------------------------------+\n"
        )
        for name, res in sorted(self.cache.items()):
            jds = [x["jd"] for x in res["prv_candidates"]]

            if res["candidate"]["jd"] > max(jds):
                latest = res["candidate"]
            else:
                latest = res["prv_candidates"][jds.index(max(jds))]

            old_flag = ""

            second_det = [x for x in jds if x > min(jds) + 0.01]

            if len(second_det) > 0:
                if Time.now().jd - second_det[0] > 1.0:
                    old_flag = "(MORE THAN ONE DAY SINCE SECOND DETECTION)"

            tns_result = " ------- "
            tns_name, tns_date, tns_group = ampel_api_tns(
                latest["ra"], latest["dec"], searchradius_arcsec=3
            )
            if tns_name:
                tns_result = tns_name

            line = "| {0} | {1} | {2:011.7f} | {3:+011.7f} | {4}      | {5:.2f} | {6:.2f}   | {7} \n".format(
                name,
                tns_result,
                float(latest["ra"]),
                float(latest["dec"]),
                ["g", "r", "i"][latest["fid"] - 1],
                latest["magpsf"],
                latest["sigmapsf"],
                old_flag,
                sign="+",
                prec=7,
            )
            table += line

        table += "+--------------------------------------------------------------------------------+\n\n"
        return table

    def draft_gcn(self):
        if self.first_obs is None:
            raise ValueError("No first observation set. Cannot draft GCN.")

        first_obs_dt = self.first_obs.datetime
        pretty_date = first_obs_dt.strftime("%Y-%m-%d")
        pretty_time = first_obs_dt.strftime("%H:%M")

        text = (
            f"Astronomer Name (Institute of Somewhere), ............. report,\n\n"
            f"On behalf of the Zwicky Transient Facility (ZTF) and Global Relay of Observatories Watching Transients Happen (GROWTH) collaborations: \n\n"
            f"As part of the ZTF neutrino follow up program (Stein et al. 2023), we observed the localization region of the {self.get_full_name()} with the Palomar 48-inch telescope, equipped with the 47 square degree ZTF camera (Bellm et al. 2019, Graham et al. 2019). {self.get_tiling_line()}"
            f"We started observations in the g- and r-band beginning at {pretty_date} {pretty_time} UTC, "
            f"approximately {(self.first_obs.jd - self.t_min.jd) * 24.0:.1f} hours after event time. {self.get_overlap_line()}"
            f"{self.get_obs_line()} \n \n"
            "The images were processed in real-time through the ZTF reduction and image subtraction pipelines at IPAC to search for potential counterparts (Masci et al. 2019). "
            "AMPEL (Nordin et al. 2019, Stein et al. 2021) was used to search the alerts database for candidates. "
            "We reject stellar sources (Tachibana and Miller 2018) and moving objects, and "
            f"apply machine learning algorithms (Mahabal et al. 2019) {self.remove_variability_line()}. "
            f"{self.get_candidates_lines()} \n\n"
        )

        if self.dist:
            text += (
                "The GW distance estimate is {:.0f} [{:.0f} - {:.0f}] Mpc.\n\n".format(
                    self.dist, self.dist - self.dist_unc, self.dist + self.dist_unc
                )
            )
        else:
            pass

        text += self.text_summary()

        text += (
            "ZTF and GROWTH are worldwide collaborations comprising Caltech, USA; IPAC, USA; WIS, Israel; OKC, Sweden; JSI/UMd, USA; DESY, Germany; TANGO, Taiwan; UW Milwaukee, USA; LANL, USA; TCD, Ireland; IN2P3, France.\n\n"
            "GROWTH acknowledges generous support of the NSF under PIRE Grant No 1545949.\n"
            "Alert distribution service provided by DIRAC@UW (Patterson et al. 2019).\n"
            "Alert database searches are done by AMPEL (Nordin et al. 2019).\n"
            "Alert filtering is performed with the nuztf (Stein et al. 2021, https://github.com/desy-multimessenger/nuztf ).\n"
        )

        return text

    @staticmethod
    def extract_ra_dec(nside: int, index: int):
        """
        Convert a healpix index to ra and dec

        :param nside: Nside of healpix map
        :param index: index of healpix pixel
        :return: ra, dec in degrees
        """
        theta, phi = hp.pix2ang(nside, index, nest=True)
        ra_deg = np.rad2deg(phi)
        dec_deg = np.rad2deg(0.5 * np.pi - theta)

        return ra_deg, dec_deg

    @staticmethod
    def extract_npix(nside: int, ra_deg: float, dec_deg: float) -> int:
        """
        Extract healpix index from ra and dec

        :param nside: nside of healpix map
        :param ra_deg: ra in degrees
        :param dec_deg: dec in degrees
        :return: index of healpix pixel
        """
        theta = 0.5 * np.pi - np.deg2rad(dec_deg)
        phi = np.deg2rad(ra_deg)

        return int(hp.ang2pix(nside, theta, phi, nest=True))

    def create_candidate_summary(self, include_ps1: bool = True) -> str:
        """
        Create pdf with lightcurve plots of all candidates

        :return None:
        """
        if len(self.cache.items()) == 0:
            self.logger.info("No candidates found, skipping pdf creation")
            return

        pdf_path = self.get_output_dir() / "candidates.pdf"

        self.logger.info(
            f"Creating overviewpdf\n(this might take a moment as it involves catalog matching)"
        )
        self.logger.debug(f"Overview pdf path: {pdf_path}")

        with PdfPages(pdf_path) as pdf:
            for name, alert in tqdm(sorted(self.cache.items())):
                fig, _ = lightcurve_from_alert(
                    [alert],
                    include_cutouts=True,
                    include_ps1=include_ps1,
                    logger=self.logger,
                    t_0_mjd=self.t_min.mjd,
                )

                pdf.savefig()
                plt.close()

    def create_overview_table(self):
        """
        Create csv table of all candidates

        :return None:
        """
        if len(self.cache.items()) == 0:
            self.logger.info("No candidates found, skipping csv creation")
            return

        csv_path = self.get_output_dir() / "candidate_table.csv"

        self.logger.info("Creating overview csv")
        self.logger.debug(f"Overview csv path: {csv_path}")

        data = {
            "ztf_id": [],
            "RA": [],
            "Dec": [],
            "mag": [],
            "xmatch": [],
            "kilonova_score": [],
        }

        for ztf_id, alert in tqdm(sorted(self.cache.items())):
            data["ztf_id"].append(ztf_id)
            data["RA"].append(alert["candidate"]["ra"])
            data["Dec"].append(alert["candidate"]["dec"])
            data["mag"].append(alert["candidate"]["magpsf"])
            data["xmatch"].append(get_cross_match_info(raw=alert, logger=self.logger))
            if alert.get("kilonova_eval") is not None:
                data["kilonova_score"].append(alert["kilonova_eval"]["kilonovaness"])
            else:
                data["kilonova_score"].append(None)

        df = pd.DataFrame.from_dict(data)

        df.to_csv(csv_path)

    def tns_summary(self):
        """
        Create summary for TNS

        :return:
        """
        summary = ""

        for name, res in sorted(self.cache.items()):
            detections = [
                x
                for x in res["prv_candidates"] + [res["candidate"]]
                if "isdiffpos" in x.keys()
            ]
            detection_jds = [x["jd"] for x in detections]
            first_detection = detections[detection_jds.index(min(detection_jds))]
            latest = [
                x
                for x in res["prv_candidates"] + [res["candidate"]]
                if "isdiffpos" in x.keys()
            ][-1]
            summary += f"Candidate: {name} / RA={res['candidate']['ra']} / Dec={res['candidate']['dec']} / First detection={first_detection['jd']}\n"
            try:
                last_upper_limit = [
                    x
                    for x in res["prv_candidates"]
                    if np.logical_and(
                        "isdiffpos" in x.keys(), x["jd"] < first_detection["jd"]
                    )
                ][-1]
                summary += f"Last Upper Limit: {last_upper_limit['jd']} / band={self.parse_ztf_filter(last_upper_limit['fid'])} / maglim={last_upper_limit['diffmaglim']:.3f}\n"

            except IndexError:
                last_upper_limit = None
                summary += "Last Upper Limit: None\n"

            summary += f"First Detection: {first_detection['jd']} / band={self.parse_ztf_filter(first_detection['fid'])} / mag={first_detection['magpsf']:.3f} +/- {first_detection['sigmapsf']:.3f}\n"

            hours_after_merger = 24.0 * (first_detection["jd"] - self.t_min.jd)

            summary += f"First observed {hours_after_merger:.2f} hours after merger\n"

            if last_upper_limit:
                summary += f"It has risen {-latest['magpsf'] + last_upper_limit['diffmaglim']} / band={self.parse_ztf_filter(latest['fid'])} / Last upper limit was in band {self.parse_ztf_filter(last_upper_limit['fid'])}\n"

            summary += f"{[x['jd'] for x in res['prv_candidates'] + [res['candidate']] if 'isdiffpos' in x.keys()]}\n"

        self.logger.info(summary)

        return summary

    def peak_mag_summary(self):
        """
        Print a summary of each candidate's peak magnitude

        :return: str
        """

        for name, res in sorted(self.cache.items()):
            detections = [
                x
                for x in res["prv_candidates"] + [res["candidate"]]
                if "isdiffpos" in x.keys()
            ]
            detection_mags = [x["magpsf"] for x in detections]
            brightest = detections[detection_mags.index(min(detection_mags))]

            diff = 0.0
            df = None

            for fid in [1, 2, 3]:
                dets = [x["magpsf"] for x in detections if int(x["fid"]) == fid]
                if len(dets) > 1:
                    nd = max(dets) - min(dets)

                    if nd > diff:
                        diff = nd
                        df = self.parse_ztf_filter(fid)

            tns_result = ""
            tns_name, tns_date, tns_group = ampel_api_tns(
                brightest["ra"], brightest["dec"], searchradius_arcsec=3.0
            )
            if tns_name:
                tns_result = f"({tns_name})"

            xmatch_info = get_cross_match_info(raw=res, logger=self.logger)

            print(
                f"Candidate {name} peaked at {brightest['magpsf']:.1f} {tns_result} "
                f"on {brightest['jd']:.1f} with "
                f"filter {self.parse_ztf_filter(brightest['fid'])}. "
                f"Max range of {diff:.1f} mag with filter {df}. {xmatch_info}"
            )

    def candidate_text(
        self, ztf_id: str, first_detection: float, lul_lim: float, lul_jd: float
    ) -> str:
        """
        Format a text summary of a candidate

        :param ztf_id: ZTF name of the candidate
        :param first_detection: first detection of the candidate
        :param lul_lim: last upper limit magnitude
        :param lul_jd: last upper limit julian date
        :return: text summary
        """
        fd = Time(first_detection, format="jd").datetime.strftime("%Y-%m-%d")

        text = f"{ztf_id} was first detected on {fd}. "

        return text

    def text_summary(self):
        """
        Create a text summary of all candidates

        :return:
        """
        text = ""
        for name, res in sorted(self.cache.items()):
            detections = [
                x
                for x in res["prv_candidates"] + [res["candidate"]]
                if "isdiffpos" in x.keys()
            ]
            detection_jds = [x["jd"] for x in detections]
            first_detection = detections[detection_jds.index(min(detection_jds))]

            latest = [
                x
                for x in res["prv_candidates"] + [res["candidate"]]
                if "isdiffpos" in x.keys()
            ][-1]

            upper_limits = [
                x for x in res["prv_candidates"] if x["jd"] < first_detection["jd"]
            ]
            if len(upper_limits) > 0:
                last_upper_limit_mag = upper_limits[-1]["diffmaglim"]
                last_upper_limit_jd = upper_limits[-1]["jd"]
            else:
                last_upper_limit_mag = None
                last_upper_limit_jd = None

            text += self.candidate_text(
                ztf_id=name,
                first_detection=first_detection["jd"],
                lul_lim=last_upper_limit_mag,
                lul_jd=last_upper_limit_jd,
            )

            ned_z, ned_dist = query_ned_for_z(
                ra_deg=latest["ra"],
                dec_deg=latest["dec"],
                searchradius_arcsec=1,
                logger=self.logger,
            )

            if ned_z:
                ned_z = float(ned_z)
                absmag = self.calculate_abs_mag(latest["magpsf"], ned_z)
                if ned_z > 0:
                    z_dist = Distance(z=ned_z, cosmology=cosmo).value
                    text += (
                        f"It has a spec-z of {ned_z:.3f} [{z_dist:.0f} Mpc] "
                        f"and an abs. mag of {absmag:.1f}. "
                        f"Distance to SDSS galaxy is {ned_dist:.2f} arcsec. "
                    )

            c = SkyCoord(res["candidate"]["ra"], res["candidate"]["dec"], unit="deg")
            g_lat = c.galactic.b.degree
            if abs(g_lat) < 15.0:
                text += f"It is located at a galactic latitude of {g_lat:.2f} degrees. "

            xmatch_info = get_cross_match_info(raw=res, logger=self.logger)
            text += xmatch_info
            text += "\n"

        if len(text) > 0:
            text = f"Amongst our candidates, \n\n{text}\n\n"

        return text

    def calculate_overlap_with_observations(
        self, first_det_window_days=3.0, min_sep=0.01, fields=None
    ):

        if fields is not None:
            new = []
            for i, field in enumerate(fields):
                new.append(
                    pd.DataFrame(
                        {
                            "filter_id": [1 for _ in range(64)],
                            "exposure_time": [300.0 for _ in range(64)],
                            "qid": [i for i in range(64)],
                            "field_id": [field for _ in range(64)],
                            "maglim": [21.0 for _ in range(64)],
                            "obsjd": [
                                self.t_min.jd + (0.02 * (i + 1)) for _ in range(64)
                            ],
                            "status": [0 for _ in range(64)],
                            "date": [
                                (self.t_min + (0.02 * (i + 1) * u.day)).isot
                                for _ in range(64)
                            ],
                        }
                    )
                )
            data = pd.concat(new)

        else:
            mns = get_obs_summary(t_min=self.t_min, max_days=first_det_window_days)

            if mns is None:
                return None, None, None

            data = mns.data.copy()

        mask = data["status"] == 0
        self.logger.info(
            f"Found {mask.sum()} successful observations in the depot, "
            f"corresponding to {np.mean(mask)*100:.2f}% of the total."
        )

        self.logger.info("Unpacking observations")

        if self.nside > 256:
            (
                self.map_coords,
                self.pixel_nos,
                self.nside,
                self.map_probs,
                self.data,
                self.total_pixel_area,
                self.key,
            ) = self.unpack_skymap(output_nside=256)

        pix_map = dict()
        pix_obs_times = dict()

        nested_pix = get_nested_pix(nside=self.nside, logger=self.logger)

        for i, obs_time in enumerate(tqdm(list(set(data["obsjd"])))):
            obs = data[data["obsjd"] == obs_time]

            field = obs["field_id"].iloc[0]

            try:
                flat_pix = nested_pix[int(field)]

                mask = obs["status"].astype(int) == 0
                indices = obs["qid"].values[mask]

                for qid in indices:
                    pixels = flat_pix[int(qid)]

                    for p in pixels:
                        if p not in pix_obs_times.keys():
                            pix_obs_times[p] = [obs_time]
                        else:
                            pix_obs_times[p] += [obs_time]

                        if p not in pix_map.keys():
                            pix_map[p] = [field]
                        else:
                            pix_map[p] += [field]

            except (KeyError, ValueError):
                self.logger.warning(
                    f"Field {field} not found in nested pix dict. "
                    f"This might be an engineering observation."
                )

        npix = hp.nside2npix(self.nside)

        ras, decs = hp.pixelfunc.pix2ang(
            self.nside, hp.nest2ring(self.nside, self.pixel_nos), lonlat=True
        )

        radecs = SkyCoord(ra=ras * u.deg, dec=decs * u.deg)

        coverage_data = []

        overlapping_fields = []
        times = []

        for i, p in enumerate(tqdm(hp.nest2ring(self.nside, self.pixel_nos))):
            entry = {
                "pixel_no": p,
                "prob": self.map_probs[i],
                "gal_b": radecs[i].galactic.b.deg,
                "in_plane": abs(radecs[i].galactic.b.deg) < 10.0,
                "ra_deg": ras[i],
                "dec_deg": decs[i],
            }

            if p in pix_obs_times.keys():
                obs = pix_obs_times[p]

                entry["n_det_class"] = 2 if max(obs) - min(obs) > min_sep else 1

                entry["latency_days"] = min(obs) - self.t_min.jd

                overlapping_fields += pix_map[p]

                times += list(obs)
            else:
                entry["n_det_class"] = 0
                entry["latency_days"] = np.nan

            coverage_data.append(entry)

        coverage_df = pd.DataFrame(coverage_data)

        overlapping_fields = sorted(list(set(overlapping_fields)))

        _observations = data.query("obsjd in @times").reset_index(drop=True)[
            ["obsjd", "exposure_time", "filter_id"]
        ]
        bands = [self.fid_to_band(fid) for fid in _observations["filter_id"].values]
        _observations["band"] = bands
        _observations.drop(columns=["filter_id"], inplace=True)
        self.observations = _observations

        self.logger.info("All observations:")
        self.logger.info(f"\n{self.observations}")

        try:
            self.first_obs = Time(min(times), format="jd")
            self.first_obs.utc.format = "isot"
            self.last_obs = Time(max(times), format="jd")
            self.last_obs.utc.format = "isot"

        except ValueError:
            err = (
                f"No observations of this event were found at any time between "
                f"{self.t_min} and {self.t_min + first_det_window_days * u.day}. "
                f"Coverage overlap is 0%!"
            )
            self.logger.error(err)
            raise ValueError(err)

        self.logger.info(f"Observations started at {self.first_obs.isot}")

        return (
            coverage_df,
            times,
            overlapping_fields,
        )

    def plot_overlap_with_observations(
        self,
        first_det_window_days=None,
        min_sep=0.01,
        fields=None,
    ):
        """
        Function to plot the overlap of the field with observations.

        :param first_det_window_days: Window of time in days to consider for the first detection.
        :param min_sep: Minimum separation between observations to consider them as separate.
        :param fields: Fields to consider.

        """

        (
            coverage_df,
            times,
            overlapping_fields,
        ) = self.calculate_overlap_with_observations(
            first_det_window_days=first_det_window_days,
            min_sep=min_sep,
            fields=fields,
        )

        if coverage_df is None:
            self.logger.warning("Not plotting overlap with observations.")
            return

        fig = plt.figure()
        plt.subplot(projection="aitoff")

        self.overlap_fields = list(set(overlapping_fields))

        overlap_mask = (coverage_df["n_det_class"] == 2) & ~coverage_df["in_plane"]
        self.overlap_prob = coverage_df[overlap_mask]["prob"].sum() * 100.0

        size = hp.max_pixrad(self.nside) ** 2 * 50.0

        veto_pixels = coverage_df.where(coverage_df["n_det_class"] == 0)

        if len(veto_pixels) > 0:
            plt.scatter(
                np.radians(veto_pixels["ra_deg"]),
                np.radians(veto_pixels["dec_deg"]),
                color="red",
                s=size,
            )

        plane_pixels = coverage_df.where(
            coverage_df["in_plane"] & (coverage_df["n_det_class"] > 0)
        )

        if len(plane_pixels) > 0:
            plt.scatter(
                np.radians(self.wrap_around_180(plane_pixels["ra_deg"])),
                np.radians(plane_pixels["dec_deg"]),
                color="green",
                s=size,
            )

        single_pixels = coverage_df.where(
            ~coverage_df["in_plane"] & (coverage_df["n_det_class"] == 1)
        )

        if len(single_pixels) > 0:
            plt.scatter(
                np.radians(self.wrap_around_180(single_pixels["ra_deg"])),
                np.radians(single_pixels["dec_deg"]),
                c=single_pixels["prob"],
                vmin=0.0,
                vmax=max(self.data[self.key]),
                s=size,
                cmap="gray",
            )

        double_pixels = coverage_df.where(
            ~coverage_df["in_plane"] & (coverage_df["n_det_class"] == 2)
        )

        if len(double_pixels) > 0:
            plt.scatter(
                np.radians(self.wrap_around_180(double_pixels["ra_deg"])),
                np.radians(double_pixels["dec_deg"]),
                c=double_pixels["prob"],
                vmin=0.0,
                vmax=max(self.data[self.key]),
                s=size,
            )

        red_patch = mpatches.Patch(color="red", label="Not observed")
        gray_patch = mpatches.Patch(color="gray", label="Observed once")
        violet_patch = mpatches.Patch(
            color="green", label="Observed Galactic Plane (|b|<10)"
        )
        plt.legend(handles=[red_patch, gray_patch, violet_patch])

        message = (
            "In total, {0:.1f} % of the contour was observed at least once.\n"
            "This estimate includes {1:.1f} % of the contour "
            "at a galactic latitude <10 deg.\n"
            "In total, {2:.1f} % of the contour was observed at least twice. \n"
            "In total, {3:.1f} % of the contour was observed at least twice, "
            "and excluding low galactic latitudes.\n"
            "These estimates account for chip gaps.".format(
                100.0 * coverage_df.query("n_det_class > 0")["prob"].sum(),
                100.0 * coverage_df.query("in_plane & n_det_class > 0")["prob"].sum(),
                100.0 * coverage_df.query("n_det_class == 2")["prob"].sum(),
                100.0 * coverage_df.query("n_det_class == 2 & ~in_plane")["prob"].sum(),
            )
        )

        n_pixels = len(coverage_df.query("n_det_class > 0 & prob > 0.0"))
        n_double = len(coverage_df.query("n_det_class == 2 & prob > 0.0"))
        n_plane = len(coverage_df.query("in_plane & n_det_class > 0 & prob > 0.0"))

        self.healpix_area = self.pixel_area * n_pixels
        self.double_extragalactic_area = self.pixel_area * n_double
        plane_area = self.pixel_area * n_plane

        self.overlap_fields = overlapping_fields

        self.logger.info(
            f"{n_pixels} pixels were covered, covering approximately "
            f"{self.healpix_area:.2g} sq deg."
        )
        self.logger.info(
            f"{n_double} pixels were covered at least twice (b>10), "
            f"covering approximately {self.double_extragalactic_area:.2g} sq deg."
        )
        self.logger.info(
            f"{n_plane} pixels were covered at low galactic latitude, "
            f"covering approximately {plane_area:.2g} sq deg."
        )
        return fig, message

    def get_exposure_summary(self) -> pd.DataFrame:
        """
        Function to get a summary of the exposures

        :return:
        """

        if self.observations is None:
            raise ValueError("No pre-observations loaded. Run plot_coverage first")

        max_days = max(self.observations["obsjd"]) - self.t_min.jd

        mns = get_obs_summary(t_min=self.t_min, max_days=max_days)

        exposures = []

        for obsjd in self.observations["obsjd"].unique():
            res = mns.data[mns.data["obsjd"] == obsjd]

            f_proc = float(len(res)) / 64.0
            f_status = np.sum(res["status"].astype(int) == 0) / 64.0

            new = {}

            copy_keys = [
                "exposure_id",
                "obsjd",
                "filter_id",
                "field_id",
                "exposure_time",
                "date",
            ]

            for key in copy_keys:
                new[key] = res[key].iloc[0]

            new["f_processed"] = f_proc
            new["f_proc_without_flag"] = f_status
            exposures.append(new)

        return pd.DataFrame(exposures)

    def get_field_summary(self):
        """
        Function to get a summary of the exposures

        :return:
        """

        df = self.get_exposure_summary()

        fields = []

        for field_entry in df["field_id"].unique():
            res = df[df["field_id"] == field_entry]

            mask = res["exposure_time"].astype(float) > 30.0

            new = {
                "field_id": field_entry,
                "n_exposures": len(res),
                "n_deep": np.sum(mask),
                "n_30": np.sum(~mask),
                "f_proc_deep": np.mean(res[mask]["f_processed"]),
                "f_proc_30": np.mean(res[~mask]["f_processed"]),
                "f_proc_without_flag_deep": np.mean(res[mask]["f_proc_without_flag"]),
                "f_proc_without_flag_30": np.mean(res[~mask]["f_proc_without_flag"]),
                "n_filters": len(res["filter_id"].unique()),
                "n_nights": len(set([int(x + 0.5) for x in res["obsjd"]])),
            }
            fields.append(new)

        summary = pd.DataFrame(fields)

        return summary.sort_values(by="field_id", ignore_index=True)

    def export_cache_to_fritz(self, group_id=None):
        if group_id is None:
            group_id = self.default_fritz_group

        saved_sources = []

        for source in self.cache.keys():
            response = save_source_to_group(object_id=source, group_id=group_id)

            if response.status_code not in [200]:
                self.logger.warning(f"Bad API call for source {source}: {response}")
            else:
                saved_sources.append(source)

        self.logger.info(f"Saved {len(saved_sources)} to fritz group {group_id}")
