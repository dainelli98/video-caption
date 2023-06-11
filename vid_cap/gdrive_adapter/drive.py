# -*- coding: utf-8 -*-
"""Functions used to manage gdrive videos."""
import io
import logging
from enum import Enum
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from httpx import AsyncClient


class FileTypeEnum(Enum):
    """Enum class for file types."""

    FOLDER = "application/vnd.google-apps.folder"
    CSV = "text/csv"
    MP4 = "video/mp4"


logger = logging.getLogger("Storage-Gdrive")


class Storage:
    """Class used to manage Google Drive storage."""

    __SCOPES = ["https://www.googleapis.com/auth/drive"]

    def __init__(self, creds_file: str) -> None:
        """Init Storage class.

        :param creds_file: Path to the credentials file.
        :type creds_file: str
        """
        self.creds_file = creds_file
        self.service = None
        self.client = AsyncClient()

    async def authenticate(self) -> None:
        """Authenticate the user to Google Drive.

        :raises e: Error during authentication:
        """
        try:
            creds = None
            if Path("token.json").exists():
                creds = Credentials.from_authorized_user_file("token.json", Storage.__SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.creds_file, Storage.__SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                with Path("token.json").open(mode="w") as token:
                    token.write(creds.to_json())

            self.service = build("drive", "v3", credentials=creds)
        except Exception as e:
            logger.error("Error during authentication: %s", e)
            raise

    async def upload_file(self, file_name: str, file_path: str) -> None:
        """Upload a file to Google Drive.

        :param file_name: File name.
        :type file_name: str
        :param file_path: Path to the file to be uploaded.
        :type file_path: str
        :raises e: Error during file upload:
        :raises e: Unexpected error during file upload:
        """
        try:
            media = MediaFileUpload(file_path)
            request = self.service.files().create(media_body=media, body={"name": file_name})
            await self.client.post(request.uri, data=request.to_json())
        except HttpError as e:
            logger.error("Error during file upload: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error during file upload: %s", e)
            raise

    async def download_file(self, file_id: str, destination: str) -> None:
        """Download a file from Google Drive.

        :param file_id: ID of the file on Google Drive to be downloaded.
        :type file_id: str
        :param destination: Path to the file to be downloaded.
        :type destination: str
        :raises e: HttpError during file download
        :raises e: Unexpected error during file download
        """
        try:
            file_content = await self.__download_file(file_id)

            with Path(destination).open(mode="wb") as f:
                f.write(file_content)
        except HttpError as e:
            logger.error("Error during file download: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error during file download: %s", e)
            raise

    async def __download_file(self, real_file_id: str) -> bytes:
        """Download a file from Google Drive.

        :param real_file_id: ID of the file on Google Drive to be downloaded.
        :type real_file_id: str
        :return: File content.
        :rtype: bytes
        """
        try:
            file_id = real_file_id

            request = self.service.files().get_media(fileId=file_id)
            file = io.BytesIO()
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                logger.info("Download %i", int(status.progress() * 100))

        except HttpError as error:
            logger.error("An error occurred: %s", error)
            file = None

        return file.getvalue()

    async def list_files(self, file_type: FileTypeEnum) -> list:
        """List all files on Google Drive.

        :param file_type: Type of files to be listed.
        :type file_type: FileTypeEnum
        :return: List of files.
        :rtype: list
        """
        try:
            # Call the Drive v3 API
            files = []
            page_token = None

            mime_type = file_type.value
            while True:
                # pylint: disable=maybe-no-member
                response = (
                    self.service.files()
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
                    path = await self.__get_full_path(file.get("id"))
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
        except HttpError as error:
            logger.error("An error occurred: %s", error)

    async def __get_full_path(self, file_id: str, path: str = "") -> str:
        """Get the full path of a file on Google Drive.

        :param file_id: ID of the file on Google Drive.
        :type file_id: str
        :param path: Path of the file on Google Drive, defaults to ""
        :type path: str, optional
        :return: Full path of the file on Google Drive.
        :rtype: str
        """
        file = (
            self.service.files()
            .get(fileId=file_id, supportsAllDrives=True, fields="id, name, parents")
            .execute()
        )
        path = file["name"] + "/" + path

        # If the file has parents, retrieve the parent's path
        if "parents" in file:
            parent_id = file["parents"][0]  # Assuming one parent
            return await self.__get_full_path(parent_id, path)
        return path

    async def close(self) -> None:
        """Close the async client session."""
        await self.client.aclose()
