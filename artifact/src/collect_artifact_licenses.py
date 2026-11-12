"""Collect upstream notices at recorded pins; no research-code license invented."""
import hashlib
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from common import CONFIG
from redraw_experiment import URLS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'third_party_licenses'


def fetch(item):
    name,repo,pin,candidates=item
    for candidate in candidates:
        url=f'https://raw.githubusercontent.com/{repo}/{pin}/{candidate}'
        try:
            with urllib.request.urlopen(url,timeout=25) as response: content=response.read()
        except urllib.error.HTTPError as error:
            if error.code==404:continue
            raise
        target=OUT/name/Path(candidate).name
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
        return {'project':name,'repository':f'https://github.com/{repo}','revision':pin,'url':url,
                'file':target.relative_to(OUT).as_posix(),'sha256':hashlib.sha256(content).hexdigest()}
    raise RuntimeError('No upstream notice found: '+name)


def main():
    candidates=['LICENSE','LICENSE.txt','LICENSE.rst','COPYING','LICENSE.md','hypothesis-python/LICENSE.txt']
    items=[(name,repo,CONFIG['pins'][name]['head'],candidates) for name,repo in URLS.items()]
    items.append(('twemoji','twitter/twemoji','v14.0.2',['LICENSE-GRAPHICS']))
    OUT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(fetch,items))
    (OUT/'sources.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'notices':len(rows),'projects':[r['project'] for r in rows]},indent=2))


if __name__=='__main__':main()
