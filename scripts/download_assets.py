"""Download versioned T-Aware research assets and verify their SHA-256 hashes."""
import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target, expected_sha256, expected_bytes):
    target = Path(target)
    if target.exists():
        if target.stat().st_size == expected_bytes and sha256(target) == expected_sha256:
            print(f"Verified cached download: {target.name}", flush=True)
            return
        raise ValueError(f"Existing download differs from the release: {target}")
    partial = target.with_name(target.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_bytes:
        raise ValueError(f"Partial download is larger than the expected file: {partial}")
    if offset < expected_bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "T-Aware-assets/1.0"})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        print(f"Downloading {target.name} ({expected_bytes / 1024**2:.1f} MiB)", flush=True)
        with urllib.request.urlopen(request, timeout=60) as response:
            resumed = offset > 0 and response.status == 206
            if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Server returned an unexpected byte range")
            with partial.open("ab" if resumed else "wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
    if partial.stat().st_size != expected_bytes or sha256(partial) != expected_sha256:
        raise ValueError(f"Download verification failed: {partial}; remove it before retrying")
    partial.rename(target)
    print(f"SHA-256 verified: {target.name}", flush=True)


def extract(archive, destination, files):
    """Extract regular files only, reject traversal, and preserve unrelated assets."""
    destination = Path(destination).resolve()
    expected = {entry["path"]: entry for entry in files}
    seen = set()
    with tarfile.open(archive, "r:gz") as package:
        members = package.getmembers()
        planned = []
        for member in members:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts or "\\" in member.name:
                raise ValueError(f"Unsafe archive entry: {member.name}")
            if member.name in seen:
                raise ValueError(f"Duplicate archive entry: {member.name}")
            seen.add(member.name)
            target = destination.joinpath(*name.parts).resolve()
            if destination not in target.parents:
                raise ValueError(f"Archive entry escapes output directory: {member.name}")
            if member.name not in expected and (not name.parts or name.parts[0] != "metadata"):
                raise ValueError(f"Archive entry is absent from the release manifest: {member.name}")
            info = expected.get(member.name)
            if info and member.size != info["bytes"]:
                raise ValueError(f"Archive entry size mismatch: {member.name}")
            if target.exists():
                if not target.is_file():
                    raise FileExistsError(target)
                if info and sha256(target) == info["sha256"]:
                    continue
                # Metadata is versioned by package; compare it before allowing reuse.
                with package.extractfile(member) as stream:
                    if sha256(target) == hashlib.sha256(stream.read()).hexdigest():
                        continue
                raise FileExistsError(f"Existing file differs from this release: {target}")
            planned.append((member, target, info))
        if set(expected) - seen:
            raise ValueError("Archive is missing files listed in its manifest")
        for member, target, info in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".extracting")
            with temporary.open("xb") as output, package.extractfile(member) as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if info and sha256(temporary) != info["sha256"]:
                raise ValueError(f"Extracted file checksum mismatch: {member.name}")
            temporary.rename(target)
    print(f"Extracted and verified: {Path(archive).name}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", default=["all"],
                        choices=["all", "temp-hc3", "robustness-benchmark", "t-aware", "classification-only"])
    parser.add_argument("--output", required=True, help="Data root for the training/evaluation commands")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "configs/assets.json")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    names = set(args.assets)
    chosen = [a for a in manifest["assets"] if "all" in names or a["id"] in names]
    if not chosen:
        parser.error("No matching release assets")
    root = Path(args.output).expanduser().resolve()
    cache = root / ".downloads" / manifest["release_tag"]
    cache.mkdir(parents=True, exist_ok=True)
    for entry in chosen:
        name = entry["filename"]
        if Path(name).name != name:
            raise ValueError("Invalid release asset filename")
        path = cache / name
        download(entry["url"], path, entry["sha256"], entry["bytes"])
        if not args.download_only:
            extract(path, root, entry["files"])
    print(f"Ready: {root}", flush=True)


if __name__ == "__main__":
    main()
