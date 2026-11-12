#!/bin/sh
# Clone the 14 subject repositories with full history and check out the HEAD commits recorded in the study
# (results/mining_stats.json), so mining is reproducible. Repositories are untrusted input: only git plumbing is
# ever executed against them. The clones (~850 MB) are not shipped; run `make data` to fetch them.
set -u
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/repos"
mkdir -p "$DEST"
while read -r r sha; do
  n=$(basename "$r")
  if [ ! -d "$DEST/$n/.git" ]; then
    for attempt in 1 2 3; do
      rm -rf "$DEST/$n"
      git clone -q "https://github.com/$r.git" "$DEST/$n" && break
      echo "retry $n ($attempt)"; sleep 5
    done
  fi
  if [ "$(git -C "$DEST/$n" rev-parse HEAD)" != "$sha" ]; then
    git -C "$DEST/$n" cat-file -e "$sha^{commit}" 2>/dev/null || git -C "$DEST/$n" fetch -q origin
    git -C "$DEST/$n" checkout -q --detach "$sha"
  fi
  echo "$n at $(git -C "$DEST/$n" rev-parse --short HEAD)"
done <<'PINS'
pytest-dev/pytest bc14038c406563b18a03e15453a10b9eee969e42
pydantic/pydantic fe5b5ce5581f10626a16dc7ff4159cf3cb6ddbec
psf/black b4f1fcae48826790b6f93ca9226d20003bce0eaa
Textualize/rich 3a109ccdffe5b2d0e30684a304fbd388e99ce57b
encode/httpx b5addb64f0161ff6bfe94c124ef76f6a1fba5254
pallets/click 8abb77a8432e03ed14a95f972b44f905279415a7
pallets/flask 49943b5a94413a530761350f04e896f76e8d620e
python-attrs/attrs 9d9f94a043de7fc4012459853a4ed597a5577855
pypa/pip 4b71f36fb4dfbc87ce65ac4710eeccf81d379f55
HypothesisWorks/hypothesis 9ab0be61aaaa27396fc1bbf4b34e7925cc68d1e2
sphinx-doc/sphinx 035a07c32c3241fd3356f4d827800c1f88525272
pylint-dev/pylint 79c8d0c9813b0c402c3890778c6d777b162dde8b
aio-libs/aiohttp d1e21bf9e819acf4b39dba3a4b0032e0e6d6410e
Textualize/textual 5e5b7ef58b8c5572c13a3ff9027d756623661495
PINS
