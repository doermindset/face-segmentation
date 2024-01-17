import os
import hashlib
import tarfile
import requests
import yaml
from tqdm import tqdm
import numpy as np


def download_and_extract_archive(url, base_folder, md5) -> None:
    """
      Downloads an archive file from a given URL, saves it to the specified base folder,
      and then extracts its contents to the base folder.

      Args:
      - url (str): The URL from which the archive file needs to be downloaded.
      - base_folder (str): The path where the downloaded archive file will be saved and extracted.
      - md5 (str): The MD5 checksum of the expected archive file for validation.
      """
    base_folder = os.path.expanduser(base_folder)
    filename = os.path.basename(url)

    download_url(url, base_folder, md5)
    archive = os.path.join(base_folder, filename)
    print(f"Extracting {archive} to {base_folder}")
    extract_tar_archive(archive, base_folder, True)


def retreive(url, save_location, chunk_size: int = 1024 * 32) -> None:
    """
        Downloads a file from a given URL and saves it to the specified location.

        Args:
        - url (str): The URL from which the file needs to be downloaded.
        - save_location (str): The path where the downloaded file will be saved.
        - chunk_size (int, optional): The size of each chunk of data to be downloaded. Defaults to 32 KB.
        """
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(save_location, 'wb') as file, tqdm(
                desc=os.path.basename(save_location),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=chunk_size):
                file.write(data)
                bar.update(len(data))

        print(f"Download successful. File saved to: {save_location}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def download_url(url: str, base_folder: str, md5: str = None) -> None:
    """Downloads the file from the url to the specified folder

    Args:
        url (str): URL to download file from
        base_folder (str): Directory to place downloaded file in
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    base_folder = os.path.expanduser(base_folder)
    filename = os.path.basename(url)
    file_path = os.path.join(base_folder, filename)

    os.makedirs(base_folder, exist_ok=True)

    # check if the file already exists
    if check_file(file_path, md5):
        print(f"File {file_path} already exists. Using that version")
        return

    print(f"Downloading {url} to file_path")
    retreive(url, file_path)

    # check integrity of downloaded file
    if not check_file(file_path, md5):
        raise RuntimeError("File not found or corrupted.")


def extract_tar_archive(from_path: str, to_path: str = None, remove_finished: bool = False) -> str:
    """Extract a tar archive.

    Args:
        from_path (str): Path to the file to be extracted.
        to_path (str): Path to the directory the file will be extracted to. If omitted, the directory of the file is
            used.
        remove_finished (bool): If True , remove the file after the extraction.
    Returns:
        (str): Path to the directory the file was extracted to.
    """
    if to_path is None:
        to_path = os.path.dirname(from_path)

    with tarfile.open(from_path, "r") as tar:
        tar.extractall(to_path)

    if remove_finished:
        os.remove(from_path)

    return to_path


def compute_md5(filepath: str, chunk_size: int = 1024 * 1024) -> str:
    with open(filepath, "rb") as f:
        md5 = hashlib.md5()
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def check_file(filepath: str, md5: str) -> bool:
    if not os.path.isfile(filepath):
        return False
    if md5 is None:
        return True
    return compute_md5(filepath) == md5


def read_params(config_path: str):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def sample_to_cuda(data):
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        data_cuda = {}
        for key in data.keys():
            data_cuda[key] = sample_to_cuda(data[key])
        return data_cuda
    elif isinstance(data, list):
        data_cuda = []
        for key in data:
            data_cuda.append(sample_to_cuda(key))
        return data_cuda
    else:
        return data.to('cuda')


def get_mask(seg_pred, class_num):
    colors = {
        0: [255, 0, 0],
        1: [0, 255, 0],
        2: [0, 0, 255]
    }

    predictions = seg_pred.argmax(dim=0).cpu().numpy()

    rgb_image = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)

    for cls in range(class_num):
        rgb_image[predictions == cls] = colors[cls]

    return rgb_image
