# -*- coding: utf-8 -*-
"""Functions used to manage gdrive videos."""
import io
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from httpx import AsyncClient
from loguru import logger


class GFileType(Enum):
    """Enum class for file types."""

    FOLDER = "application/vnd.google-apps.folder"
    CSV = "text/csv"
    MP4 = "video/mp4"


class Storage:
    """Class used to manage Google Drive storage.

    :param creds_file: Path to the credentials file.
    """

    _SCOPES: ClassVar[list[str]] = ["https://www.googleapis.com/auth/drive"]

    _creds_file: str
    _service: Any
    _client: AsyncClient

    def __init__(self, creds_file: str) -> None:
        self._creds_file = creds_file
        self._service = None
        self._client = AsyncClient()

    async def authenticate(self) -> None:
        """Authenticate the user to Google Drive."""
        creds = None
        if Path("token.json").exists():
            creds = Credentials.from_authorized_user_file("token.json", Storage._SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self._creds_file, Storage._SCOPES)
                creds = flow.run_local_server(port=0)

            with Path("token.json").open(mode="w") as token:
                token.write(creds.to_json())

        self._service = build("drive", "v3", credentials=creds)

    async def upload_file(self, file_name: str, file_path: str) -> None:
        """Upload a file to Google Drive.

        :param file_name: File name.
        :param file_path: Path to the file to be uploaded.
        """
        media = MediaFileUpload(file_path)
        request = self._service.files().create(media_body=media, body={"name": file_name})
        await self._client.post(request.uri, data=request.to_json())

    async def download_file(self, file_id: str, destination: str) -> None:
        """Download a file from Google Drive.

        :param file_id: ID of the file on Google Drive to be downloaded.
        :param destination: Path to the file to be downloaded.
        """
        file_content = await self._download_file(file_id)

        with Path(destination).open(mode="wb") as f:
            f.write(file_content)

    async def _download_file(self, real_file_id: str) -> bytes:
        """Download a file from Google Drive.

        :param real_file_id: ID of the file on Google Drive to be downloaded.
        :return: File content.
        """
        try:
            file_id = real_file_id

            request = self._service.files().get_media(fileId=file_id)
            file = io.BytesIO()
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                logger.info("Download %i", int(status.progress() * 100))

        except HttpError as exc:
            logger.error("An error occurred: %s", exc)
            file = None

        return file.getvalue()

    async def list_files(self, file_type: GFileType) -> list[str]:
        """List all files on Google Drive.

        :param file_type: Type of files to be listed.
        :return: List of files.
        """
        files = []
        page_token = None

        mime_type = file_type.value
        while True:
            response = (
                self._service.files()
                .list(
                    q=f"mimeType='{mime_type}'" if mime_type else None,
                    corpora="drive",
                    supportsAllDrives=True,
                    driveId="0AEdryJTRFjjbUk9PVA",
                    includeItemsFromAllDrives=True,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, size)",
                    pageToken=page_token,
                )
                .execute()
            )
            for file in response.get("files", []):
                path = await self._get_full_path(file.get("id"))
                logger.info(path)
                logger.info(
                    "Found file: %s, %s (%s)",
                    file.get("name"),
                    file.get("id"),
                    file.get("size"),
                )

            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken", None)
            if page_token is None:
                break
        return files

    async def _get_full_path(self, file_id: str, path: str = "") -> str:
        """Get the full path of a file on Google Drive.

        :param file_id: ID of the file on Google Drive.
        :param path: Path of the file on Google Drive, defaults to ""
        :return: Full path of the file on Google Drive.
        """
        file = (
            self._service.files()
            .get(fileId=file_id, supportsAllDrives=True, fields="id, name, parents")
            .execute()
        )
        path = file["name"] + "/" + path

        # If the file has parents, retrieve the parent's path
        if "parents" in file:
            parent_id = file["parents"][0]  # Assuming one parent
            return await self._get_full_path(parent_id, path)
        return path

    async def close(self) -> None:
        """Close the async client session."""
        await self._client.aclose()
