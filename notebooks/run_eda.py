#!/usr/bin/env python3
"""Reproduce stage-one EDA only. Run: python notebooks/run_eda.py [--skip-auxiliary]."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import os

# Keep plotting cache outside read-only home directories.
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cs5228-eda-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/cs5228-eda-cache')
from eda_core import run as run_core
from eda_figures import run as run_figures

def hashes(paths, root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--skip-auxiliary',action='store_true',help='Only regenerate core tables and figures; report not regenerated.')
    args=parser.parse_args();root=args.root.resolve()
    output=root/'experiments'/'eda';output.mkdir(parents=True,exist_ok=True)
    source_paths=sorted((root/'data').rglob('*.csv'))
    protected=[root/'proposal'/'main.md',root/'proposal'/'slides.md']+sorted((root/'final report').rglob('*'))
    protected=[p for p in protected if p.is_file()]
    before=hashes(source_paths+protected,root)
    summary,clean=run_core(root,output/'core')
    run_figures(clean,output/'core',root/'proposal'/'figures')
    if not args.skip_auxiliary:
        from eda_auxiliary import run as run_auxiliary
        run_auxiliary(root,output/'auxiliary')
    after=hashes(source_paths+protected,root)
    assert before==after,'A protected input or report changed during EDA.'
    manifest={'python':platform.python_version(),'executable':sys.executable,
              'packages':{p:importlib.metadata.version(p) for p in ['pandas','numpy','matplotlib']},
              'input_sha256':{k:v for k,v in before.items() if k.startswith('data/')},
              'protected_unchanged':before==after,'protected_sha256':{k:v for k,v in before.items() if not k.startswith('data/')},
              'code_sha256':hashes(sorted((root/'notebooks').glob('*.py')),root),
              'scope':'Stage 1 only; test has no target; no model fitting, predictions or submissions',
              'command':'python notebooks/run_eda.py'+(' --skip-auxiliary' if args.skip_auxiliary else '')}
    (output/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    if not args.skip_auxiliary:
        from eda_report import write_report
        write_report(root)
    print(json.dumps({'status':'complete','train_rows':summary['rows']['train'],'test_rows':summary['rows']['test'],
                      'protected_unchanged':True,'outputs':str(output)},ensure_ascii=False))

if __name__=='__main__': main()
