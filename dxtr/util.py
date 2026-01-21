from google.cloud import storage
from dotenv import load_dotenv
import os
import json

from dxtr import constants


def upload_to_gcs(source_file_path, destination_blob_name):
    """
    Uploads a file to a Google Cloud Storage bucket.

    :param bucket_name: Name of your GCS bucket (e.g., 'my-bucket')
    :param source_file_path: Local path to the file (e.g., 'local/path/test.md')
    :param destination_blob_name: Path within the bucket (e.g., 'profiles/test.md')
    """
    # Initialize the client
    storage_client = storage.Client()

    # Get the bucket object
    bucket = storage_client.bucket(constants.blob_store_root)

    # Create a blob object and upload the file
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)

    print(f"File {source_file_path} uploaded to {destination_blob_name}.")


def read_from_gcs(blob_name: str) -> str:
    """
    Reads a file from a Google Cloud Storage bucket as a string.

    :param blob_name: Path within the bucket (e.g., 'profiles/test.md')
    :return: Content of the file as a string.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(constants.blob_store_root)
    blob = bucket.blob(blob_name)
    return blob.download_as_text()


def read_json_from_gcs(blob_name: str) -> dict | list:
    """
    Reads a JSON file from a Google Cloud Storage bucket.

    :param blob_name: Path within the bucket (e.g., 'profiles/data.json')
    :return: Parsed JSON content.
    """
    content = read_from_gcs(blob_name)
    return json.loads(content)


def listdir_gcs(prefix: str) -> list[str]:
    """
    List contents of a GCS 'directory' (prefix).

    Works like os.listdir() - returns names of blobs and common prefixes
    within the specified prefix.
    """
    # Initialize the client
    storage_client = storage.Client()

    # Get the bucket object
    bucket = storage_client.bucket(constants.blob_store_root)

    # Ensure prefix ends with a slash to treat it as a directory
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    # Use delimiter='/' to group results by "directory"
    blobs = bucket.list_blobs(prefix=prefix, delimiter="/")

    results = []
    # We must iterate to populate prefixes and blobs
    for blob in blobs:
        # Get only the name relative to the prefix
        relative_name = blob.name[len(prefix) :]
        if relative_name:
            results.append(relative_name)

    # Add "subdirectories" (common prefixes)
    for subdir in blobs.prefixes:
        # subdir is 'prefix/dirname/'
        relative_subdir = subdir[len(prefix) :].rstrip("/")
        if relative_subdir:
            results.append(relative_subdir)

    return results
