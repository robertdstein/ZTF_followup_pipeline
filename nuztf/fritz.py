import os

import backoff
import requests  # type: ignore

from nuztf.credentials import load_credentials

# Fritz API URLs

API_BASEURL = "https://fritz.science"

fritz_token = load_credentials("fritz", token_based=True)


def fritz_api(method: str, endpoint_extension: str, data: dict = None):
    """
    Make a request to the Fritz API

    :param method: Method to use
    :param endpoint_extension: Endpoint extension (e.g. "api/sources")
    :param data: Data to send (e.g. {"ra": 0, "dec": 0})
    :return:
    """
    headers = {"Authorization": f"token {fritz_token}"}
    endpoint = os.path.join(API_BASEURL, endpoint_extension)
    if method in ["post", "POST"]:
        response = requests.request(method, endpoint, json=data, headers=headers)
    elif method in ["get", "GET"]:
        response = requests.request(method, endpoint, params=data, headers=headers)
    else:
        raise ValueError("You have to use either 'get' or 'post'")
    return response


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.RequestException,
    max_time=60,
)
def save_source_to_group(object_id: str, group_id: int):
    payload = {
        "objId": object_id,
        "inviteGroupIds": [group_id],
    }
    return fritz_api(
        method="POST", endpoint_extension="api/source_groups", data=payload
    )


def delete_source_from_group(object_id: str, group_id: int):
    payload = {
        "objId": object_id,
        "unsaveGroupIds": [group_id],
    }
    return fritz_api(
        method="POST", endpoint_extension="api/source_groups", data=payload
    )
