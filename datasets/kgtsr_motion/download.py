"""Download, verify and extract the public blurred-only dataset release."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile


RELEASE = 'https://github.com/shenzexin/KG-TSRDeblur/releases/download/dataset-v1.0.0'


def download(url, path):
    request = urllib.request.Request(url, headers={'User-Agent': 'KGTSR-dataset-downloader/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response, path.open('wb') as output:
        shutil.copyfileobj(response, output, 1024*1024)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(path, output):
    root = output.resolve()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            target = (root/info.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Archive path escapes output directory')
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Symbolic links are not accepted')
        archive.extractall(root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='data')
    p.add_argument('--archives', default='data/downloads')
    a = p.parse_args()
    manifest = json.loads(Path(__file__).with_name('release_assets.json').read_text(encoding='utf-8'))
    output, archives = Path(a.output), Path(a.archives)
    output.mkdir(parents=True, exist_ok=True)
    archives.mkdir(parents=True, exist_ok=True)
    for asset in manifest['assets']:
        local = archives/asset['name']
        if not local.exists() or digest(local) != asset['sha256']:
            print('Downloading '+asset['name'], flush=True)
            temporary = local.with_suffix(local.suffix+'.partial')
            download(RELEASE+'/'+asset['name'], temporary)
            if digest(temporary) != asset['sha256']:
                raise ValueError('SHA-256 mismatch for '+asset['name'])
            temporary.replace(local)
        if local.suffix == '.zip':
            safe_extract(local, output)
        print('Verified '+asset['name'], flush=True)
    print('Dataset root: '+str(output/manifest['archive_root']))


if __name__ == '__main__':
    main()
