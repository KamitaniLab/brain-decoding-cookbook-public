import os
import shutil
import stat
import argparse
import json
import urllib.request
import hashlib
import zipfile
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
                    unzip_file(output, dest)


def unzip_file(archive: str, extract_dir: str) -> None:
    '''Extract a zip archive, skipping members that are already extracted.

    A member whose extracted file already exists with the same size is left
    untouched. Re-running a target, or running another target that shares the
    archive, therefore neither rewrites the files nor fails on ones that were
    extracted read-only (e.g., by the `unzip` command, which keeps the
    archive's permission bits).
    '''
    n_extracted, n_skipped = 0, 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename

            # don't extract absolute paths or ones with .. in them
            # (same rule as shutil.unpack_archive)
            if name.startswith('/') or '..' in name:
                continue

            target = os.path.join(extract_dir, *name.split('/'))
            if info.is_dir():
                os.makedirs(target, exist_ok=True)
                continue

            if os.path.isfile(target) and os.path.getsize(target) == info.file_size:
                n_skipped += 1
                continue

            # A partial or stale file may be read-only; make it writable
            # before replacing it
            if os.path.exists(target):
                os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
                os.remove(target)

            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(info) as fsrc, open(target, 'wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)
            n_extracted += 1

    print(f'  {n_extracted} extracted, {n_skipped} already present')


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