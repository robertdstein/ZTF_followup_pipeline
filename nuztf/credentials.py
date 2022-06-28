#!/usr/bin/env python
# coding: utf-8

import os
import warnings
import logging
from ztfquery import io

# Manage ztfquery logins from environment variables


def load_credentials(name):
    """ZTFquery wrapper for loading credentials."""
    return io._load_id_(name)


try:
    io.set_account(
        "irsa", username=os.environ["IRSA_USER"], password=os.environ["IRSA_PASSWORD"]
    )
    logging.info('Set up "irsa" credentials')

except KeyError:
    logging.info(
        'No Credentials for "irsa" found in environment' "Assuming they are set."
    )

try:
    io.set_account(
        "skyvision",
        username=os.environ["SKYVISION_USER"],
        password=os.environ["SKYVISION_PASSWORD"],
    )
    logging.info('Set up "skyvision" credentials')

except KeyError:
    logging.info(
        'No Credentials for "skyvision" found in environment' "Assuming they are set."
    )

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        io.set_account(
            "ampel_api_archive_token",
            username=os.environ["AMPEL_API_ARCHIVE_TOKEN_USER"],
            password=os.environ["AMPEL_API_ARCHIVE_TOKEN_PASSWORD"],
        )
        logging.info('Set up "ampel_api_archive_token" credentials')

except KeyError:
    logging.info("No Token for AMPEL API found in environment" "Assume they are set.")

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        io.set_account(
            "tns_api_token",
            username=os.environ["TNS_API_TOKEN_USER"],
            password=os.environ["TNS_API_TOKEN_PASSWORD"],
        )
        logging.info('Set up "tns_api_token" credentials')

except KeyError:
    logging.info("No Token for TNS API found in environment" "Assume they are set.")

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        io.set_account(
            "ipac_fp_global",
            username=os.environ["IPAC_FP_GLOBAL_USER"],
            password=os.environ["IPAC_FP_GLOBAL_PASSWORD"],
        )
        logging.info('Set up "ipac_fp_global" credentials')
except KeyError:
    logging.info(
        "No global IPAC forced photometry account found in environment"
        "Assume they are set."
    )

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        io.set_account(
            "ipac_fp_personal",
            username=os.environ["IPAC_FP_PERSONAL_EMAIL"],
            password=os.environ["IPAC_FP_PERSONAL_PASSWORD"],
        )
        logging.info('Set up "tns_api_token" credentials')

except KeyError:
    logging.info(
        "No personal IPAC forced photometry account found in environment"
        "Assume they are set."
    )
