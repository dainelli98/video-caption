# -*- coding: utf-8 -*-
import io
import pickle
from pathlib import Path

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from httpx import AsyncClient


class Storage:
    """Storage class that encapsulates Google Drive API operations.

    Attributes
    ----------
        creds_file (str): Path to the Google Drive API credentials file.
        service (obj): Service object for the Google Drive API.
        client (obj): AsyncClient object for asynchronous operations.
    """

    def __init__(self, creds_file) -> None:
        """Constructs all the necessary attributes for the Storage object.

        Args:
        ----
            creds_file (str): Path to the Google Drive API credentials file.
        """
        self.creds_file = creds_file
        self.service = None
        self.client = AsyncClient()

    async def authenticate(self):
        """Authenticates the client with Google Drive."""
        try:
            creds = None
            if Path.exists("token.pickle"):
                with open("token.pickle", "rb") as token:
                    creds = pickle.load(token)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.creds_file, ["https://www.googleapis.com/auth/drive"]
                    )
                    creds = flow.run_local_server(port=0)

                with open("token.pickle", "wb") as token:
                    pickle.dump(creds, token)

            self.service = build("drive", "v3", credentials=creds)
        except Exception as e:
            print(f"Error during authentication: {e}")
            raise e

    async def upload_file(self, file_name, file_path):
        """Uploads a file to Google Drive.

        Args:
        ----
            file_name: Name of the file on Google Drive after upload.
            file_path: Path to the local file to be uploaded.
        """
        try:
            media = MediaFileUpload(file_path)
            request = self.service.files().create(media_body=media, body={"name": file_name})
            await self.client.post(request.uri, data=request.to_json())
        except HttpError as e:
            print(f"Error during file upload: {e}")
            raise e
        except Exception as e:
            print(f"Unexpected error during file upload: {e}")
            raise e

    async def download_file(self, file_id, destination):
        """Ownloads a file from Google Drive.

        Args:
        ----
            file_id: ID of the file on Google Drive to be downloaded.
            destination: Local path where the downloaded file will be saved.
        """
        try:
            file_content = await self.__download_file(file_id)

            with open(destination, "wb") as f:
                f.write(file_content)
        except HttpError as e:
            print(f"Error during file download: {e}")
            raise e
        except Exception as e:
            print(f"Unexpected error during file download: {e}")
            raise e

    async def __download_file(self, real_file_id):
        """Downloads a file from Google Drive (internal use).

        Args:
        ----
            real_file_id: ID of the file on Google Drive to be downloaded.
        """
        try:
            file_id = real_file_id

            request = self.service.files().get_media(fileId=file_id)
            file = io.BytesIO()
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}.")

        except HttpError as error:
            print(f"An error occurred: {error}")
            file = None

        return file.getvalue()

    async def list_files(self, file_type):
        """Lists files on Google Drive based on the specified file type.

        Args:
        ----
            file_type: Type of the file (e.g., 'folders', 'csv', 'mp4').
        """
        try:
            # Call the Drive v3 API
            files = []
            page_token = None
            mime_types = {
                "folders": "application/vnd.google-apps.folder",
                "csv": "text/csv",
                "mp4": "video/mp4",
            }
            mime_type = mime_types.get(file_type)
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
                    print(path)
                    print(f'Found file: {file.get("name")}, {file.get("id")} ({file.get("size")})')
                files.extend(response.get("files", []))
                page_token = response.get("nextPageToken", None)
                if page_token is None:
                    break
            return files
        except HttpError as error:
            print(f"An error occurred: {error}")
        except HttpError as error:
            print(f"An error occurred: {error}")

    async def __get_full_path(self, file_id, path=""):
        """Retrieves the full path of a file on Google Drive.

        Args:
        ----
            file_id: ID of the file on Google Drive.
            path: Path of the file (optional).
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

    async def close(self):
        """Closes the async client session."""
        await self.client.aclose()
