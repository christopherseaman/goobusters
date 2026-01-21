"""
Minimal MD.ai client for iOS - sync/download functionality only.
Based on mdai==0.16.0, stripped of heavy dependencies (pydicom, numpy, etc.)
"""

import os
import threading
import re
import json
import uuid
import zipfile
import requests
import urllib3.exceptions
from retrying import retry
from tqdm import tqdm
import arrow


def retry_on_http_error(exception):
    valid_exceptions = [
        requests.exceptions.HTTPError,
        requests.exceptions.ConnectionError,
        urllib3.exceptions.HTTPError,
    ]
    return any([isinstance(exception, e) for e in valid_exceptions])


class Client:
    """Client for communicating with MD.ai backend API.
    Communication is via user access tokens (in MD.ai Hub, Settings -> User Access Tokens).
    """

    def __init__(self, domain="public.md.ai", access_token=None):
        domain_match = re.match(r"^\w+\.md\.ai$", domain)
        dev_domain_match = re.match(r"^\w+\.mdai.dev(:\d+)?$", domain)
        if not domain_match and not dev_domain_match:
            raise ValueError(f"domain {domain} is invalid: should be format *.md.ai")

        self.domain = domain
        self.access_token = access_token
        self.session = requests.Session()
        self._test_endpoint()

    def project(
        self,
        project_id,
        dataset_id=None,
        label_group_id=None,
        path=".",
        force_download=False,
        annotations_only=False,
        extract_images=True,
    ):
        """Download project data from MD.ai.
        
        Returns a simple object with annotations_fp and images_dir attributes.
        """
        if path == ".":
            print("Using working directory for data.")
        else:
            os.makedirs(path, exist_ok=True)
            print(f"Using path '{path}' for data.")

        data_manager_kwargs = {
            "domain": self.domain,
            "project_id": project_id,
            "dataset_id": dataset_id,
            "label_group_id": label_group_id,
            "path": path,
            "session": self.session,
            "headers": self._create_headers(),
            "force_download": force_download,
            "extract_images": extract_images,
        }

        annotations_data_manager = ProjectDataManager("annotations", **data_manager_kwargs)
        annotations_data_manager.create_data_export_job()
        if not annotations_only:
            images_data_manager = ProjectDataManager("images", **data_manager_kwargs)
            images_data_manager.create_data_export_job()

        annotations_data_manager.wait_until_ready()
        if not annotations_only:
            images_data_manager.wait_until_ready()
            # Return simple object with paths instead of full Project class
            return SimpleProject(
                annotations_fp=annotations_data_manager.data_path,
                images_dir=images_data_manager.data_path,
            )
        else:
            print("No project created. Downloaded annotations only.")
            return None

    def _create_headers(self):
        headers = {}
        if self.access_token:
            headers["x-access-token"] = self.access_token
        return headers

    def _test_endpoint(self):
        """Checks endpoint for validity and authorization."""
        test_endpoint = f"https://{self.domain}/api/test"
        r = self.session.get(test_endpoint, headers=self._create_headers())
        if r.status_code == 200:
            print(f"Successfully authenticated to {self.domain}.")
        else:
            raise Exception("Authorization error. Make sure your access token is valid.")


class SimpleProject:
    """Minimal project class - just holds paths."""
    def __init__(self, annotations_fp, images_dir):
        self.annotations_fp = annotations_fp
        self.images_dir = images_dir


class ProjectDataManager:
    """Manager for project data exports and downloads."""

    def __init__(
        self,
        data_type,
        domain=None,
        project_id=None,
        dataset_id=None,
        label_group_id=None,
        model_id=None,
        format=None,
        path=".",
        session=None,
        headers=None,
        force_download=False,
        extract_images=True,
    ):
        if data_type not in ["images", "annotations", "model-outputs", "dicom-metadata"]:
            raise ValueError(
                "data_type must be 'images', 'annotations', 'model-outputs' or 'dicom-metadata'."
            )
        if not domain:
            raise ValueError("domain is not specified.")
        if not project_id:
            raise ValueError("project_id is not specified.")
        if not os.path.exists(path):
            raise OSError(f"Path '{path}' does not exist.")

        self.data_type = data_type
        self.force_download = force_download
        self.extract_images = extract_images

        self.domain = domain
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.label_group_id = label_group_id
        self.format = format
        self.model_id = model_id
        self.path = path
        if session and isinstance(session, requests.Session):
            self.session = session
        else:
            self.session = requests.Session()
        self.headers = headers

        # path for downloaded data
        self.data_path = None
        # ready threading event
        self._ready = threading.Event()

    def create_data_export_job(self):
        """Create data export job through MD.ai API."""
        endpoint = f"https://{self.domain}/api/data-export/{self.data_type}"
        params = self._get_data_export_params()
        r = self.session.post(endpoint, json=params, headers=self.headers)
        if r.status_code == 202:
            msg = f"Preparing {self.data_type} export for project {self.project_id}..."
            print(msg.ljust(100))
            self._check_data_export_job_progress()
        else:
            if r.status_code == 401:
                msg = (
                    f"Project {self.project_id} at domain {self.domain}"
                    + " does not exist or you do not have sufficient permissions for access."
                )
                print(msg)
            self._on_data_export_job_error()

    def wait_until_ready(self):
        self._ready.wait()

    def _get_data_export_params(self):
        if self.data_type == "images":
            params = {
                "projectHashId": self.project_id,
                "datasetHashId": self.dataset_id,
                "exportFormat": "zip",
            }
        elif self.data_type == "annotations":
            params = {
                "projectHashId": self.project_id,
                "datasetHashId": self.dataset_id,
                "labelGroupHashId": self.label_group_id,
                "exportFormat": "json",
            }
        elif self.data_type == "model-outputs":
            params = {
                "projectHashId": self.project_id,
                "datasetHashId": self.dataset_id,
                "modelHashId": self.model_id,
                "exportFormat": "json",
            }
        elif self.data_type == "dicom-metadata":
            params = {
                "projectHashId": self.project_id,
                "datasetHashId": self.dataset_id,
                "exportFormat": self.format,
            }
        return params

    @retry(
        retry_on_exception=retry_on_http_error,
        wait_exponential_multiplier=100,
        wait_exponential_max=1000,
        stop_max_attempt_number=10,
    )
    def _check_data_export_job_progress(self):
        """Poll for data export job progress."""
        endpoint = f"https://{self.domain}/api/data-export/{self.data_type}/progress"
        params = self._get_data_export_params()
        r = self.session.post(endpoint, json=params, headers=self.headers)
        if r.status_code != 200:
            r.raise_for_status()

        try:
            body = r.json()
            status = body["status"]
        except (TypeError, KeyError):
            self._on_data_export_job_error()
            return

        if status == "done":
            self._on_data_export_job_done()

        elif status == "error":
            self._on_data_export_job_error()

        elif status == "running":
            try:
                progress = int(body["progress"])
            except (TypeError, ValueError):
                progress = 0
            try:
                time_remaining = int(body["timeRemaining"])
            except (TypeError, ValueError):
                time_remaining = 0

            # print formatted progress info
            if time_remaining > 45:
                time_remaining_fmt = (
                    arrow.now().shift(seconds=time_remaining).humanize(only_distance=True)
                )
            else:
                time_remaining_fmt = f"{time_remaining} seconds"
            end_char = "\r" if progress < 100 else "\n"
            msg = (
                f"Exporting {self.data_type} for project {self.project_id}..."
                + f"{progress}% (time remaining: {time_remaining_fmt})."
            )
            print(msg.ljust(100), end=end_char, flush=True)

            # run progress check at 1s intervals so long as status == 'running'
            t = threading.Timer(1.0, self._check_data_export_job_progress)
            t.start()

    @retry(
        retry_on_exception=retry_on_http_error,
        wait_exponential_multiplier=100,
        wait_exponential_max=1000,
        stop_max_attempt_number=10,
    )
    def _on_data_export_job_done(self):
        endpoint = f"https://{self.domain}/api/data-export/{self.data_type}/done"
        params = self._get_data_export_params()
        r = self.session.post(endpoint, json=params, headers=self.headers)
        if r.status_code != 200:
            r.raise_for_status()

        try:
            file_keys = r.json()["fileKeys"]

            if file_keys:
                data_path = self._get_data_path(file_keys)
                if self.force_download or not os.path.exists(data_path):
                    # download in separate thread
                    t = threading.Thread(target=self._download_files, args=(file_keys,))
                    t.start()
                else:
                    # use existing data
                    self.data_path = data_path
                    print(f"Using cached {self.data_type} data for project {self.project_id}.")
                    # fire ready threading.Event
                    self._ready.set()
        except (TypeError, KeyError):
            self._on_data_export_job_error()

    @retry(
        retry_on_exception=retry_on_http_error,
        wait_exponential_multiplier=100,
        wait_exponential_max=1000,
        stop_max_attempt_number=10,
    )
    def _on_data_export_job_error(self):
        endpoint = f"https://{self.domain}/api/data-export/{self.data_type}/error"
        params = self._get_data_export_params()
        r = self.session.post(endpoint, json=params, headers=self.headers)
        if r.status_code != 200:
            r.raise_for_status()
        print(f"Error exporting {self.data_type} for project {self.project_id}.")
        # fire ready threading.Event
        self._ready.set()

    def _get_data_path(self, file_keys):
        if self.data_type == "images":
            # should be folder for zip file
            images_dir = re.sub(r"(_part\d+of\d+)?\.\S+$", "", file_keys[0])
            return os.path.join(self.path, images_dir)
        elif self.data_type == "annotations":
            annotations_fp = file_keys[0]
            return os.path.join(self.path, annotations_fp)
        elif self.data_type == "model-outputs":
            model_outputs_fp = file_keys[0]
            return os.path.join(self.path, model_outputs_fp)
        elif self.data_type == "dicom-metadata":
            dicom_metadata_fp = file_keys[0]
            return os.path.join(self.path, dicom_metadata_fp)

    def _download_files(self, file_keys):
        """Downloads exported files."""
        try:
            for file_key in file_keys:
                print(f"Downloading file: {file_key}")
                filepath = os.path.join(self.path, file_key)

                key = requests.utils.quote(file_key)
                dl_session_id = str(uuid.uuid4())

                # request download token
                url = f"https://{self.domain}/api/data-export/download-request"
                data = {"key": key, "sessionId": dl_session_id}
                r = requests.post(url, json=data, headers=self.headers)
                dl_token = r.json().get("token")

                # download file
                url = f"https://{self.domain}/api/data-export/download/{key}"
                data = {"token": dl_token, "sessionId": dl_session_id}
                r = requests.post(url, json=data, stream=True)
                # fallback to GET if POST not available
                if r.status_code == 405:
                    r = requests.get(url, params=data, stream=True)

                # total size in bytes
                total_size = int(r.headers.get("content-length", 0))
                block_size = 32 * 1024
                wrote = 0
                with open(filepath, "wb") as f:
                    with tqdm(
                        total=total_size, unit="B", unit_scale=True, unit_divisor=1024
                    ) as pbar:
                        for chunk in r.iter_content(block_size):
                            f.write(chunk)
                            wrote = wrote + len(chunk)
                            pbar.update(block_size)
                if total_size != 0 and wrote != total_size:
                    raise IOError(f"Error downloading file {file_key}.")

                if self.data_type == "images" and self.extract_images:
                    # unzip archive
                    print(f"Extracting archive: {file_key}")
                    with zipfile.ZipFile(filepath, "r") as f:
                        f.extractall(self.path)

            self.data_path = self._get_data_path(file_keys)

            print(f"Success: {self.data_type} data for project {self.project_id} ready.")
        except Exception:
            print(f"Error downloading {self.data_type} data for project {self.project_id}.")

        # fire ready threading.Event
        self._ready.set()
