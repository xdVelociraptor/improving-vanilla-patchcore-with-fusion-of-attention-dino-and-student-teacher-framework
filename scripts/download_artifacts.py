"""Restore checksum-pinned research artifacts using Python's standard library."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(root, name):
    name = name.replace('\\', '/')
    parts = PurePosixPath(name)
    if parts.is_absolute() or '..' in parts.parts or ':' in name:
        raise ValueError('Unsafe archive path: ' + name)
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Archive path escapes destination')
    return target


def write_member(stream, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an existing research result or configuration.
    if target.exists():
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
        if digest(target) != h.hexdigest():
            raise RuntimeError('Different existing file; use a fresh checkout: ' + str(target))
        return
    temp = target.with_name(target.name + '.download-part')
    try:
        with temp.open('wb') as out:
            shutil.copyfileobj(stream, out)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def extract(archive, entry, root):
    base = safe_path(root, entry.get('destination', '.'))
    if entry['format'] == 'file':
        with archive.open('rb') as stream:
            write_member(stream, base)
    elif entry['format'] == 'zip':
        with zipfile.ZipFile(archive) as z:
            if entry.get('member') and entry['member'] not in z.namelist():
                raise ValueError('Required archive member is missing: ' + entry['member'])
            for member in z.infolist():
                if member.is_dir():
                    continue
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Archive symlink rejected')
                if entry.get('member') and member.filename != entry['member']:
                    continue
                target = safe_path(base, member.filename)
                with z.open(member) as stream:
                    write_member(stream, target)
    elif entry['format'] == 'tar':
        with tarfile.open(archive) as tar:
            for member in tar:
                # Only the original Imagenette train split is required.
                if not member.name.startswith('imagenette2/train/'):
                    continue
                target = safe_path(base, member.name)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError('Archive link or special file rejected')
                with tar.extractfile(member) as stream:
                    write_member(stream, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--groups', nargs='+', default=['datasets', 'models', 'scores', 'outputs', 'weights', 'imagenette'])
    parser.add_argument('--plan', action='store_true', help='List downloads without making changes')
    parser.add_argument('--keep-downloads', action='store_true')
    args = parser.parse_args()
    entries = json.loads((ROOT / 'artifact-downloads.json').read_text())
    unknown = set(args.groups) - {e['group'] for e in entries}
    if unknown:
        parser.error('Unknown groups: ' + ', '.join(sorted(unknown)))
    cache = args.root / '.artifact-downloads'
    for entry in entries:
        if entry['group'] not in args.groups:
            continue
        print(entry['group'], entry['name'], flush=True)
        if args.plan:
            continue
        cache.mkdir(parents=True, exist_ok=True)
        archive = safe_path(cache, entry['name'])
        if not archive.exists() or digest(archive) != entry['sha256']:
            partial = archive.with_suffix(archive.suffix + '.part')
            req = urllib.request.Request(entry['url'], headers={'User-Agent': 'PatchCore-artifact-setup'})
            with urllib.request.urlopen(req, timeout=120) as response, partial.open('wb') as out:
                shutil.copyfileobj(response, out)
            if digest(partial) != entry['sha256']:
                raise RuntimeError('Checksum mismatch: ' + entry['name'])
            partial.replace(archive)
        extract(archive, entry, args.root)
        if not args.keep_downloads:
            archive.unlink()
        print('Verified and restored:', entry['name'], flush=True)
    print('Download plan complete.' if args.plan else 'Selected artifacts restored successfully.')


if __name__ == '__main__':
    main()
