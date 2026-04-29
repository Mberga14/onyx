"""Unit tests for GCS authentication methods in the blob connector."""

import json
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from onyx.connectors.blob.connector import BlobStorageConnector
from onyx.connectors.exceptions import ConnectorValidationError
from onyx.connectors.models import ConnectorMissingCredentialError


class TestBlobGCSAuthentication:
    """Test GCS authentication methods in BlobStorageConnector."""

    def _make_connector(self) -> BlobStorageConnector:
        return BlobStorageConnector(
            bucket_type="google_cloud_storage",
            bucket_name="test-gcs-bucket",
        )

    def test_gcs_load_credentials_hmac(self) -> None:
        """Test that existing HMAC path works with explicit authentication_method."""
        connector = self._make_connector()

        with patch("boto3.client") as mock_boto3:
            mock_s3_client = Mock()
            mock_boto3.return_value = mock_s3_client

            connector.load_credentials(
                {
                    "authentication_method": "access_key",
                    "access_key_id": "test-hmac-key",
                    "secret_access_key": "test-hmac-secret",
                }
            )

            # Should use boto3 S3 client with GCS endpoint
            mock_boto3.assert_called_once()
            call_kwargs: dict[str, Any] = mock_boto3.call_args[1]
            assert call_kwargs["endpoint_url"] == "https://storage.googleapis.com"
            assert call_kwargs["aws_access_key_id"] == "test-hmac-key"
            assert call_kwargs["aws_secret_access_key"] == "test-hmac-secret"
            assert connector.s3_client == mock_s3_client
            assert connector._gcs_native_client is None

    def test_gcs_load_credentials_hmac_default_method(self) -> None:
        """Test that HMAC is the default when authentication_method is absent."""
        connector = self._make_connector()

        with patch("boto3.client") as mock_boto3:
            mock_boto3.return_value = Mock()

            connector.load_credentials(
                {
                    "access_key_id": "test-hmac-key",
                    "secret_access_key": "test-hmac-secret",
                }
            )

            mock_boto3.assert_called_once()
            assert connector.s3_client is not None
            assert connector._gcs_native_client is None

    def test_gcs_load_credentials_service_account(self) -> None:
        """Test service account JSON authentication creates native GCS client."""
        connector = self._make_connector()
        mock_client_instance = Mock()
        mock_credentials = Mock()

        sa_info = {"type": "service_account", "project_id": "my-project"}

        with (
            patch(
                "google.oauth2.service_account.Credentials.from_service_account_info",
                return_value=mock_credentials,
            ) as mock_from_info,
            patch(
                "google.cloud.storage.Client",
                return_value=mock_client_instance,
            ) as mock_client_cls,
        ):
            connector.load_credentials(
                {
                    "authentication_method": "service_account",
                    "service_account_json": json.dumps(sa_info),
                }
            )

            mock_from_info.assert_called_once()
            mock_client_cls.assert_called_once_with(credentials=mock_credentials, project="my-project")
            assert connector._gcs_native_client == mock_client_instance
            assert connector.s3_client is None

    def test_gcs_load_credentials_adc(self) -> None:
        """Test ADC/Workload Identity authentication creates native GCS client."""
        connector = self._make_connector()
        mock_client_instance = Mock()

        with patch(
            "google.cloud.storage.Client",
            return_value=mock_client_instance,
        ) as mock_client_cls:
            connector.load_credentials({"authentication_method": "adc"})

            mock_client_cls.assert_called_once_with()
            assert connector._gcs_native_client == mock_client_instance
            assert connector.s3_client is None

    def test_gcs_load_credentials_invalid_method(self) -> None:
        """Test that invalid authentication method raises error."""
        connector = self._make_connector()

        with pytest.raises(ConnectorValidationError, match="Invalid authentication"):
            connector.load_credentials({"authentication_method": "invalid_method"})

    def test_gcs_load_credentials_hmac_missing_keys(self) -> None:
        """Test that missing HMAC keys raises error."""
        connector = self._make_connector()

        with pytest.raises(ConnectorMissingCredentialError):
            connector.load_credentials(
                {
                    "authentication_method": "access_key",
                    "access_key_id": "test-key",
                    # missing secret_access_key
                }
            )

    def test_gcs_load_credentials_sa_missing_json(self) -> None:
        """Test that missing SA JSON raises error."""
        connector = self._make_connector()

        with pytest.raises(ConnectorMissingCredentialError):
            connector.load_credentials(
                {
                    "authentication_method": "service_account",
                    # missing service_account_json
                }
            )


class TestBlobGCSNativeDispatch:
    """Test that methods dispatch correctly when native GCS client is set."""

    def test_download_object_dispatches_to_gcs(self) -> None:
        """Test that _download_object uses native GCS when available."""
        connector = BlobStorageConnector(
            bucket_type="google_cloud_storage",
            bucket_name="test-bucket",
        )

        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_as_bytes.return_value = b"file content"
        mock_blob.size = 100

        connector._gcs_native_client = mock_client
        connector.size_threshold = None

        result = connector._download_object("path/to/file.txt")

        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.blob.assert_called_once_with("path/to/file.txt")
        mock_blob.download_as_bytes.assert_called_once()
        assert result == b"file content"

    def test_download_object_gcs_respects_size_threshold(self) -> None:
        """Test that GCS download skips files exceeding size threshold."""
        connector = BlobStorageConnector(
            bucket_type="google_cloud_storage",
            bucket_name="test-bucket",
        )

        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.size = 999_999_999  # Very large file

        connector._gcs_native_client = mock_client
        connector.size_threshold = 1024

        result = connector._download_object("path/to/huge-file.bin")

        assert result is None
        mock_blob.download_as_bytes.assert_not_called()

    def test_validate_gcs_native(self) -> None:
        """Test GCS native validation calls list_blobs."""
        connector = BlobStorageConnector(
            bucket_type="google_cloud_storage",
            bucket_name="test-bucket",
            prefix="my-prefix/",
        )

        mock_client = Mock()
        mock_bucket = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.list_blobs.return_value = iter([])

        connector._gcs_native_client = mock_client

        # Should not raise
        connector._validate_gcs_native()

        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.list_blobs.assert_called_once_with(prefix="my-prefix/", max_results=1)
