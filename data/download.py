import os
import shutil
import argparse
import json

from bdpy.dataset.utils import download_file, download_splitted_file


def main(cfg):
    with open(cfg.filelist, 'r') as f:
        filelist = json.load(f)

    target = filelist[cfg.target]

    for fl in target['files']:
        output = os.path.join(target['save_in'], fl['name'])

        # Downloading
        if not os.path.exists(output):
            if 'splitted' in fl and fl['splitted']:
                print(f'Downloading {output}')
                download_splitted_file(fl['split_files'], output, progress_bar=True, md5sum=fl['md5sum'])
            else:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', default='files.json')
    parser.add_argument('target')

    cfg = parser.parse_args()

    main(cfg)
