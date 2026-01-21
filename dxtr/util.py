from google.cloud import storage
from google.cloud.exceptions import NotFound
from dotenv import load_dotenv
import os
import json
import asyncio

from dxtr import constants


async def upload_to_gcs(source_file_path, destination_blob_name):
    """
    Uploads a file to a Google Cloud Storage bucket.

    :param bucket_name: Name of your GCS bucket (e.g., 'my-bucket')
    :param source_file_path: Local path to the file (e.g., 'local/path/test.md')
    :param destination_blob_name: Path within the bucket (e.g., 'profiles/test.md')
    """
    def _upload():
        # Initialize the client
        storage_client = storage.Client()

        # Get the bucket object
        bucket = storage_client.bucket(constants.blob_store_root)

        # Create a blob object and upload the file
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path)

        print(f"File {source_file_path} uploaded to {destination_blob_name}.")

    await asyncio.to_thread(_upload)


async def read_from_gcs(blob_name: str) -> str:
    """
    Reads a file from a Google Cloud Storage bucket as a string.

    :param blob_name: Path within the bucket (e.g., 'profiles/test.md')
    :return: Content of the file as a string.
    """
    def _read():
        storage_client = storage.Client()
        bucket = storage_client.bucket(constants.blob_store_root)
        blob = bucket.blob(blob_name)
        return blob.download_as_text()

    try:
        return await asyncio.to_thread(_read)
    except NotFound:
        print(f"File {blob_name} not found in GCS.")
        return ""


async def read_json_from_gcs(blob_name: str) -> dict | list:
    """
    Reads a JSON file from a Google Cloud Storage bucket.

    :param blob_name: Path within the bucket (e.g., 'profiles/data.json')
    :return: Parsed JSON content.
    """
    content = await read_from_gcs(blob_name)
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {blob_name}")
        return {}


async def listdir_gcs(prefix: str) -> list[str]:
    """
    List contents of a GCS 'directory' (prefix).

    Works like os.listdir() - returns names of blobs and common prefixes
    within the specified prefix.
    """
    def _list():
        # Initialize the client
        storage_client = storage.Client()

        # Get the bucket object
        bucket = storage_client.bucket(constants.blob_store_root)
        
        # Ensure prefix ends with a slash to treat it as a directory
        # (Need local variable because we can't modify the outer scope 'prefix' directly in a closure nicely, 
        # though strictly speaking python strings are immutable so we just make a new one)
        local_prefix = prefix
        if local_prefix and not local_prefix.endswith("/"):
            local_prefix += "/"

        # Use delimiter='/' to group results by "directory"
        blobs = bucket.list_blobs(prefix=local_prefix, delimiter="/")

        results = []
        # We must iterate to populate prefixes and blobs
        for blob in blobs:
            # Get only the name relative to the prefix
            relative_name = blob.name[len(local_prefix) :]
            if relative_name:
                results.append(relative_name)

        # Add "subdirectories" (common prefixes)
        for subdir in blobs.prefixes:
            # subdir is 'prefix/dirname/'
            relative_subdir = subdir[len(local_prefix) :].rstrip("/")
            if relative_subdir:
                results.append(relative_subdir)

        return results

    return await asyncio.to_thread(_list)
