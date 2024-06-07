from minio import Minio


class MinioClient:
    def __init__(
            self,
            endpoint: str,
            access_key: str = None,
            secret_key: str = None
    ) -> None:
        self.endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key

        self.client = Minio(
            endpoint=self.endpoint,
            access_key=self._access_key,
            secret_key=self._secret_key,
            secure=False,
        )

    def send_to_bucket(
            self,
            bucket_name: str,
            file: str,
            directory: str = None,
            name_on_storage: str = None,
    ) -> None:
        """Sends a file to an existing bucket"""

        if name_on_storage is None:
            name_on_storage = str(file).split("/")[-1]
        if directory is not None:
            name_on_storage = f"{directory}/{name_on_storage}"
        self.client.fput_object(
            bucket_name=bucket_name,
            object_name=name_on_storage,
            file_path=file
        )
