This is a Python-based library that simplifies interactions with Google Drive using `gdrive_adapter`. It provides a simple interface for authenticating, listing files, uploading and downloading files.

## Requirements

- Python 3.7+
- `gdrive_adapter` module

### Usage example:

NOTE: Authentication in order to interact with Google Drive, you need to authenticate with a ``credentials.json`` file that contains your Google Drive API credentials.


```python
import asyncio
from gdrive_adapter import drive

async def main():

    library = drive.Storage('{PROJECT_CONFIG_PATH}/credentials.json')

    #Initiates the authentication process.
    await library.authenticate()

    # List files
    # file_type: Type of the file (e.g., 'folders', 'csv', 'mp4'
    files = await library.list_files(file_type="folders")

    # Upload a file
    await library.upload_file('filename_on_gdrive', '/path/to/test.txt')

    # Download a file
    await library.download_file('file_id', './local_path/to/filename.xyz')

    #close the async client session
    await library.close()

if __name__ == '__main__':
    asyncio.run(main())
```
