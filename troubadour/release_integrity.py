"""Hash exactly the publishable files, excluding local caches and this manifest."""
import argparse
import hashlib
import json
from pathlib import Path

MANIFEST = 'SHA256SUMS.json'


def file_hashes(folder):
    folder=Path(folder);out={}
    for path in sorted(folder.rglob('*')):
        relative=path.relative_to(folder)
        if not path.is_file() or str(relative)==MANIFEST:
            continue
        if any(part.startswith('.') or part=='__pycache__' for part in relative.parts):
            continue
        out[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def write_manifest(folder):
    values=file_hashes(folder)
    (Path(folder)/MANIFEST).write_text(json.dumps(values,indent=2)+'\n')
    return values


def verify_manifest(folder):
    expected=json.loads((Path(folder)/MANIFEST).read_text())
    actual=file_hashes(folder)
    if actual!=expected:
        missing=sorted(expected.keys()-actual.keys())
        extra=sorted(actual.keys()-expected.keys())
        changed=sorted(k for k in expected.keys()&actual.keys() if expected[k]!=actual[k])
        raise ValueError({'missing':missing,'extra':extra,'changed':changed})
    return len(actual)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True)
    parser.add_argument('--write',action='store_true');args=parser.parse_args()
    if args.write:write_manifest(args.folder)
    print(json.dumps({'folder':args.folder,'verified_files':verify_manifest(args.folder)}))

if __name__=='__main__':main()
