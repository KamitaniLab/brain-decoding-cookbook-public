import os
import shutil
import argparse
import json
import urllib.request
import hashlib
from typing import Union

from tqdm import tqdm


def main(cfg):
    with open(cfg.filelist, 'r') as f:
        filelist = json.load(f)

    target = filelist[cfg.target]

    for fl in target['files']:
        output = os.path.join(target['save_in'], fl['name'])

        # Downloading
        if not os.path.exists(output):
            print(f'Downloading {output} from {fl["url"]}')
            download_file(fl['url'], output, progress_bar=True, md5sum=fl['md5sum'])

        # Postprocessing
        if 'postproc' in fl:
            for pp in fl['postproc']:
                if pp['name'] == 'unzip':
                    print(f'Unzipping {output}')
                    if 'destination' in pp:
                        dest = pp['destination']
                    else:
                        dest = './'
                    shutil.unpack_archive(output, extract_dir=dest)


def download_file(url: str, destination: str, progress_bar: bool = True, md5sum: Union[str, None] = None) -> None:
    '''Download a file.'''

    response = urllib.request.urlopen(url)
    file_size = int(response.info()["Content-Length"])

    def _show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            progress_bar.update(downloaded - progress_bar.n)

    with tqdm(total=file_size, unit='B', unit_scale=True, desc=destination, ncols=100) as progress_bar:
        urllib.request.urlretrieve(url, destination, _show_progress)

    if md5sum is not None:
        md5_hash = hashlib.md5()
        with open(destination, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5_hash.update(chunk)
        md5sum_test = md5_hash.hexdigest()
        if md5sum != md5sum_test:
            raise ValueError(f'md5sum mismatch. \nExpected: {md5sum}\nActual: {md5sum_test}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', default='files.json')
    parser.add_argument('target')

    cfg = parser.parse_args()

    main(cfg)
