"""
iOS-compatible MD.ai client using httpx (no pandas, numpy, or opencv).

Minimal implementation that provides the same API as the mdai SDK but uses
only pure Python data structures (dict/list) and httpx for HTTP calls.

Dependencies: httpx only (already vendored)
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

import httpx


class MDaiClient:
    """
    Lightweight MD.ai REST API client using httpx.
    Replaces mdai.Client for iOS compatibility.
    """

    def __init__(self, domain: str, access_token: str, timeout: float = 120.0):
        """
        Initialize MD.ai client.

        Args:
            domain: MD.ai domain (e.g., "ucsf.md.ai")
            access_token: MD.ai API access token
            timeout: Request timeout in seconds
        """
        self.domain = domain
        self.access_token = access_token
        self.timeout = timeout
        self.headers = {
            "x-access-token": access_token,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """
        Make HTTP request to MD.ai API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for httpx.request
                     - json: JSON body (for POST)
                     - params: Query parameters (for GET)
                     - stream: Stream response (for downloads)
                     - no_auth: If True, don't send MD.ai auth headers (for external redirects)

        Returns:
            HTTP response object

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        url = f"https://{self.domain}{endpoint}"
        
        # Don't send MD.ai auth headers for external redirects (like Google Cloud Storage)
        # The signed URL is self-contained and doesn't need our token
        no_auth = kwargs.pop("no_auth", False)
        if not no_auth:
        kwargs.setdefault("headers", self.headers)
        else:
            # Only send Content-Type, not auth headers
            kwargs.setdefault("headers", {"Content-Type": "application/json"})
        
        kwargs.setdefault("timeout", self.timeout)

        # Handle streaming requests differently - httpx uses stream() method, not stream kwarg
        use_stream = kwargs.pop("stream", False)
        
        # httpx follows redirects by default, but explicitly enable it
        # When following redirects to external URLs (like Google Cloud Storage),
        # httpx should not forward Authorization headers automatically
        with httpx.Client(follow_redirects=True) as client:
            if use_stream:
                # For streaming, use stream() context manager
                with client.stream(method, url, **kwargs) as response:
                    response.raise_for_status()
                    # Return the streaming response - caller must iterate over it
                    return response
            else:
            response = client.request(method, url, **kwargs)
            response.raise_for_status()
            return response

    def _poll_export_job(
        self, endpoint: str, params: dict, poll_interval: float = 2.0
    ) -> None:
        """
        Poll export job until completion.

        Args:
            endpoint: Base endpoint for the export job (e.g., "/api/data-export/images")
            params: Parameters to send with progress check (projectHashId, datasetHashId, etc.)
            poll_interval: Seconds between polling attempts

        Raises:
            RuntimeError: If export job fails
        """
        progress_endpoint = f"{endpoint}/progress"
        while True:
            # MD.ai API uses POST for progress checks, not GET
            status_resp = self._request("POST", progress_endpoint, json=params)
            body = status_resp.json()
            status = body.get("status")

            if status == "done":
                return
            elif status == "error":
                error_msg = body.get("message", "Unknown error")
                raise RuntimeError(f"MD.ai export job failed: {error_msg}")

            # Show progress if available
            if status == "running":
                progress = body.get("progress", 0)
                time_remaining = body.get("timeRemaining", 0)
                if progress:
                    print(f"  Progress: {progress}%", end="\r")
                if time_remaining and time_remaining > 0:
                    print(f"  Estimated time remaining: {time_remaining}s", end="\r")

            time.sleep(poll_interval)

    def project(self, project_id: str, dataset_id: str, path: str) -> "ProjectResult":
        """
        Download project data (annotations + images/videos) from MD.ai.
        Drop-in compatible with mdai.Client.project() - uses same API flow.

        Args:
            project_id: MD.ai project hash ID
            dataset_id: MD.ai dataset hash ID
            path: Local directory to save data

        Returns:
            ProjectResult object with images_dir attribute

        Raises:
            RuntimeError: If download fails
        """
        import uuid
        import re
        
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        def _download_data_type(data_type: str, export_format: str) -> str:
            """Download a data type (annotations or images) using SDK's exact flow."""
            endpoint = f"/api/data-export/{data_type}"
            params = {
                "projectHashId": project_id,
                "datasetHashId": dataset_id,
                "exportFormat": export_format,
            }

        # Create export job
            resp = self._request("POST", endpoint, json=params)
            if resp.status_code != 202:
                raise RuntimeError(
                    f"Failed to create {data_type} export job: HTTP {resp.status_code}"
                )

        # Poll for completion
            self._poll_export_job(endpoint, params)
            
            # Get fileKeys from /done endpoint (same as SDK)
            done_resp = self._request("POST", f"{endpoint}/done", json=params)
            done_data = done_resp.json()
            file_keys = done_data.get("fileKeys", [])
            
            if not file_keys:
                raise RuntimeError(f"No file keys returned for {data_type} export")
            
            # Download each file using SDK's method
            for file_key in file_keys:
                print(f"Downloading {data_type} file: {file_key}")
                filepath = path_obj / file_key
                
                # URL-encode the key
                from urllib.parse import quote
                key_encoded = quote(file_key)
                dl_session_id = str(uuid.uuid4())

                # Request download token (same as SDK)
                token_resp = self._request(
                    "POST",
                    "/api/data-export/download-request",
                    json={"key": file_key, "sessionId": dl_session_id}
                )
                dl_token = token_resp.json().get("token")
                
                # Download file (try POST first, fallback to GET like SDK)
                # Note: SDK doesn't send headers to download endpoint - the token/sessionId in body/params is enough
                # When following redirects to Google Cloud Storage, don't send MD.ai auth headers
                download_url = f"/api/data-export/download/{key_encoded}"
                try:
                    download_resp = self._request(
                        "POST",
                        download_url,
                        json={"token": dl_token, "sessionId": dl_session_id},
                        stream=True,
                        no_auth=True  # Don't send MD.ai headers - redirect to GCS doesn't need them
                    )
                    # Streaming response - iterate over chunks
                    with open(filepath, "wb") as f:
                        for chunk in download_resp.iter_bytes(chunk_size=32 * 1024):
                            f.write(chunk)
                except Exception as e:
                    # Fallback to GET if POST returns 405 (like SDK does)
                    try:
        download_resp = self._request(
                            "GET",
                            download_url,
                            params={"token": dl_token, "sessionId": dl_session_id},
                            stream=True,
                            no_auth=True  # Don't send MD.ai headers - redirect to GCS doesn't need them
                        )
                        # Streaming response - iterate over chunks
                        with open(filepath, "wb") as f:
                            for chunk in download_resp.iter_bytes(chunk_size=32 * 1024):
                                f.write(chunk)
                    except Exception:
                        raise RuntimeError(f"Failed to download file {file_key}: {e}")
                
                # Extract if it's a zip (images)
                if data_type == "images" and filepath.suffix == ".zip":
                    print(f"Extracting archive: {file_key}")
                    # Extract to directory (remove .zip extension)
                    extract_dir = path_obj / re.sub(r"(_part\d+of\d+)?\.zip$", "", file_key)
                    with zipfile.ZipFile(filepath, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                    filepath.unlink()  # Remove zip after extraction
                    return str(extract_dir)
            
            # For annotations, return the file path
            if data_type == "annotations":
                return str(path_obj / file_keys[0])
            # For images, return the extracted directory
            images_dir = re.sub(r"(_part\d+of\d+)?\.zip$", "", file_keys[0])
            return str(path_obj / images_dir)

        # Download annotations
        print(f"Downloading annotations for project {project_id}, dataset {dataset_id}...")
        annotations_path = _download_data_type("annotations", "json")
        print(f"✓ Annotations saved: {Path(annotations_path).name}")

        # Download images (videos)
        print(f"Downloading images/videos...")
        images_dir = _download_data_type("images", "zip")
        print(f"✓ Images extracted to: {Path(images_dir).name}")

        return ProjectResult(images_dir)


class ProjectResult:
    """Simple container for project download results."""

    def __init__(self, images_dir: str):
        self.images_dir = images_dir

    def set_labels_dict(self, labels_dict: dict[str, int]) -> None:
        """Compatibility method (no-op for iOS client)."""
        pass


def json_to_dataframe(
    json_file: str, datasets: list[str] | None = None
) -> dict[str, Any]:
    """
    Load and parse MD.ai annotations JSON file.

    Returns dict with 'annotations', 'studies', and 'labels' as plain lists/dicts
    (no pandas DataFrames).

    Args:
        json_file: Path to MD.ai annotations JSON file
        datasets: Optional list of dataset IDs to filter (default: all)

    Returns:
        Dictionary with keys:
            - annotations: List of annotation dicts
            - studies: List of study dicts
            - labels: List of label dicts
    """
    if datasets is None:
        datasets = []

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_annotations = []
    all_studies = []

    # Process each dataset
    for dataset in data.get("datasets", []):
        dataset_id = dataset.get("id", "")
        dataset_name = dataset.get("name", "Unknown")

        # Filter by dataset IDs if specified
        if datasets and dataset_id not in datasets:
            continue

        # Add dataset context to studies
        for study in dataset.get("studies", []):
            study["dataset"] = dataset_name
            study["datasetId"] = dataset_id
            all_studies.append(study)

        # Add dataset context to annotations
        for annot in dataset.get("annotations", []):
            annot["dataset"] = dataset_name
            annot["datasetId"] = dataset_id
            all_annotations.append(annot)

    # Process labels from label groups
    all_labels = []
    for label_group in data.get("labelGroups", []):
        group_id = label_group.get("id")
        group_name = label_group.get("name")

        for label in label_group.get("labels", []):
            label_entry = {
                "labelGroupId": group_id,
                "labelGroupName": group_name,
                "annotationMode": label.get("annotationMode"),
                "color": label.get("color"),
                "description": label.get("description"),
                "labelId": label.get("id"),
                "labelName": label.get("name"),
                "radlexTagIdsLabel": label.get("radlexTagIds"),
                "scope": label.get("scope"),
            }

            # Add parentId if it exists
            if "parentId" in label:
                label_entry["parentLabelId"] = label.get("parentId")

            all_labels.append(label_entry)

    return {
        "annotations": all_annotations,
        "studies": all_studies,
        "labels": all_labels,
    }


# ============================================================================
# Utility functions for working with plain dict/list data
# ============================================================================


def filter_annotations(annotations: list[dict], label_ids: list[str]) -> list[dict]:
    """
    Filter annotations by label IDs.

    Args:
        annotations: List of annotation dicts
        label_ids: List of label IDs to keep

    Returns:
        Filtered list of annotations
    """
    return [ann for ann in annotations if ann.get("labelId") in label_ids]


def group_by_video(
    annotations: list[dict],
) -> dict[tuple[str, str], list[dict]]:
    """
    Group annotations by (StudyInstanceUID, SeriesInstanceUID).

    Args:
        annotations: List of annotation dicts

    Returns:
        Dictionary mapping (study_uid, series_uid) to list of annotations
    """
    grouped = {}
    for ann in annotations:
        key = (ann.get("StudyInstanceUID"), ann.get("SeriesInstanceUID"))
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(ann)
    return grouped


def add_video_paths(annotations: list[dict], base_dir: str) -> list[dict]:
    """
    Add video_path field to each annotation.

    Args:
        annotations: List of annotation dicts
        base_dir: Base directory containing videos

    Returns:
        Annotations with video_path field added (modifies in place)
    """
    from pathlib import Path

    for ann in annotations:
        study_uid = ann.get("StudyInstanceUID")
        series_uid = ann.get("SeriesInstanceUID")
        video_path = Path(base_dir) / study_uid / f"{series_uid}.mp4"
        ann["video_path"] = str(video_path)
        ann["file_exists"] = video_path.exists()

    return annotations


def sort_annotations_by_frame(annotations: list[dict]) -> list[dict]:
    """
    Sort annotations by frame number.

    Args:
        annotations: List of annotation dicts

    Returns:
        Sorted list of annotations
    """
    return sorted(annotations, key=lambda x: x.get("frameNumber", 0))
