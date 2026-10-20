#!/usr/bin/env python3
"""Check a pinned source export in a temporary tree; no model or GPU calls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def run(*argv, cwd=None):
    return subprocess.run(argv, cwd=cwd, capture_output=True, check=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Existing llama.cpp Git checkout containing the pinned base object')
    parser.add_argument('--compile-policy', action='store_true')
    parser.add_argument('--manifest', default='manifest.json',
                        choices=('manifest.json', 'manifest-39da130f.json'),
                        help='Select the historical or latest pinned source generation')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root / 'patches' / 'llama.cpp'
    manifest = json.loads((directory / args.manifest).read_text())
    patch = directory / manifest['patch']
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == manifest['patch_sha256']
    with tempfile.TemporaryDirectory(prefix='native-patch-check-') as temporary:
        tree = Path(temporary)
        for name, expected in manifest['files'].items():
            relative = Path(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            target = tree / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if expected['before_sha256'] is not None:
                data = run('git', '-C', str(args.source), 'show', manifest['base_commit'] + ':' + name)
                assert hashlib.sha256(data).hexdigest() == expected['before_sha256'], name
                target.write_bytes(data)
        run('git', 'apply', '--check', '--whitespace=error', str(patch), cwd=tree)
        run('git', 'apply', str(patch), cwd=tree)
        for name, expected in manifest['files'].items():
            assert hashlib.sha256((tree / name).read_bytes()).hexdigest() == expected['after_sha256'], name
        if args.compile_policy:
            executable = tree / 'test-prefix-selection'
            run('c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                str(tree / 'tests/test-server-prefix-selection.cpp'), '-o', str(executable))
            print(run(str(executable)).decode().strip())
    print(f"Pinned patch applies; all {len(manifest['files'])} source digests match the export.")


if __name__ == '__main__':
    main()
