#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Read source hashes and apply the hook patch only inside a temporary tree."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tabby_checkout',type=Path)
    parser.add_argument('--exl-root',type=Path,help='Optional installed exllamav3 package directory; never imported')
    args=parser.parse_args();meta=json.loads((ROOT/'PROVENANCE.json').read_text())
    head=subprocess.check_output(['git','-C',str(args.tabby_checkout),'rev-parse','HEAD'],text=True).strip()
    if head!=meta['upstream']['commit']:raise SystemExit('Tabby commit differs from the tested base')
    for row in meta['hooks']:
        if sha(args.tabby_checkout/row['path'])!=row['base_sha256']:raise SystemExit('Tabby base source differs: '+row['path'])
    for row in meta['modules']:
        if sha(ROOT/'overlay'/row['path'])!=row['exported_sha256']:raise SystemExit('Export module differs: '+row['path'])
    if args.exl_root:
        for row in meta['native_dependency']['checked_files']:
            if sha(args.exl_root/row['path'])!=row['sha256']:raise SystemExit('Native source differs from qualified v1.6.3: '+row['path'])
    patch=ROOT.parent.parent/'patches/tabby/0001-exl3-persistent-context-hooks.patch'
    with tempfile.TemporaryDirectory(prefix='tabby-prefix-compat-') as directory:
        root=Path(directory)
        for row in meta['hooks']:
            target=root/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(args.tabby_checkout/row['path'],target)
        subprocess.run(['git','apply','--check',str(patch)],cwd=root,check=True)
        subprocess.run(['git','apply',str(patch)],cwd=root,check=True)
        for row in meta['hooks']:
            path=root/row['path']
            if sha(path)!=row['tested_sha256']:raise SystemExit('Patch output differs from tested source: '+row['path'])
            compile(path.read_bytes(),str(path),'exec')
        for path in (ROOT/'overlay/common').glob('*.py'):compile(path.read_bytes(),str(path),'exec')
    print(json.dumps({'status':'compatible','tabby_base':head,'hook_files':len(meta['hooks']),
                     'module_files':len(meta['modules']),'native_checked':bool(args.exl_root),
                     'checkout_modified':False,'model_imported':False,'inference_calls':0}))

if __name__=='__main__':main()
