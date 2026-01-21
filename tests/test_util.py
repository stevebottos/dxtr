import json
from unittest.mock import MagicMock, patch
from dxtr.util import read_from_gcs, read_json_from_gcs

@patch("dxtr.util.storage.Client")
@patch("dxtr.util.constants")
def test_read_from_gcs(mock_constants, mock_storage_client):
    mock_constants.blob_store_root = "test-bucket"
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_storage_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    mock_blob.download_as_text.return_value = "hello world"

    content = read_from_gcs("test-path")

    assert content == "hello world"
    mock_bucket.blob.assert_called_once_with("test-path")
    mock_blob.download_as_text.assert_called_once()

@patch("dxtr.util.read_from_gcs")
def test_read_json_from_gcs(mock_read_from_gcs):
    mock_read_from_gcs.return_value = json.dumps({"key": "value"})

    data = read_json_from_gcs("test-path.json")

    assert data == {"key": "value"}
    mock_read_from_gcs.assert_called_once_with("test-path.json")
