"""Run on GitHub Actions: verify curated assets, then publish an atomic draft release."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def gh(*args):
    return subprocess.check_output(['gh', *args], text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-config', required=True)
    args = parser.parse_args()
    settings = json.loads(Path(args.source_config).read_text())
    if not settings.get('enabled', True):
        print('No asset publication requested')
        return
    repository = 'twzydn20000928/T-Aware'
    assert os.environ['GITHUB_REPOSITORY'] == repository
    source = settings['source_url'].rstrip('/')
    assert source.startswith('https://')
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        manifest_file = folder / 'assets-manifest.json'
        request = Request(source + '/assets-manifest.json', headers={'User-Agent': 'T-Aware-release'})
        with urlopen(request, timeout=60) as response:
            raw = response.read(4 * 1024 * 1024)
        assert hashlib.sha256(raw).hexdigest() == settings['manifest_sha256']
        manifest_file.write_bytes(raw)
        manifest = json.loads(raw)
        assert manifest['repository'] == repository
        tag = manifest['release_tag']
        assert re.fullmatch(r'v\d+\.\d+\.\d+', tag)
        sums = ''.join(a['sha256'] + '  ' + a['filename'] + '\n' for a in manifest['assets'])
        sums += digest(manifest_file) + '  assets-manifest.json\n'
        (folder / 'SHA256SUMS').write_bytes(sums.encode('utf-8'))
        expected = {a['filename']: {'bytes': a['bytes'], 'sha256': a['sha256']}
                    for a in manifest['assets']}
        for path in [manifest_file, folder / 'SHA256SUMS']:
            expected[path.name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        # Draft releases are resolved through the CLI and their numeric API URL.
        # GitHub's releases/tags endpoint can return 404 until a draft is published.
        query = subprocess.run(['gh', 'release', 'view', tag, '--repo', repository,
                                '--json', 'isDraft,apiUrl'], text=True, capture_output=True)
        existing, remote = None, {}
        if query.returncode == 0:
            existing = json.loads(query.stdout)
            if not existing['isDraft']:
                raise RuntimeError('Release already published; refusing to replace immutable assets')
            draft = json.loads(gh('api', existing['apiUrl']))
            remote = {a['name']: a for a in draft['assets']}
        def matches(name):
            entry = remote.get(name, {})
            return (entry.get('state') == 'uploaded' and entry.get('size') == expected[name]['bytes']
                    and entry.get('digest') == 'sha256:' + expected[name]['sha256'])
        names = set()
        for asset in manifest['assets']:
            name = asset['filename']
            assert re.fullmatch(r'[a-zA-Z0-9._-]+\.tar\.gz', name) and name not in names
            names.add(name)
            assert asset['bytes'] < 2 * 1024**3 and re.fullmatch(r'[a-f0-9]{64}', asset['sha256'])
            if matches(name):
                print('Verified existing release asset', name, flush=True)
                continue
            path = folder / name
            print('Downloading', name, asset['bytes'], flush=True)
            subprocess.run(['curl', '--fail', '--location', '--retry', '5', '--retry-all-errors',
                            '--connect-timeout', '30', '--max-time', '2400', '--silent', '--show-error',
                            '--output', str(path), source + '/' + name], check=True)
            assert path.stat().st_size == asset['bytes'], name
            assert digest(path) == asset['sha256'], name
            print('Verified', name, flush=True)
        if existing is None:
            gh('release', 'create', tag, '--repo', repository, '--target', os.environ['GITHUB_SHA'],
               '--draft', '--title', 'T-Aware & Temp-HC3 ' + tag, '--notes-file', 'docs/release-notes.md')
            existing = json.loads(gh('release', 'view', tag, '--repo', repository, '--json', 'apiUrl,isDraft'))
        uploads = [str(p) for p in folder.iterdir() if not matches(p.name)]
        if uploads:
            gh('release', 'upload', tag, '--repo', repository, '--clobber', *uploads)
        published = json.loads(gh('api', existing['apiUrl']))
        remote = {a['name']: a for a in published['assets']}
        for name, entry in remote.items():
            assert entry['state'] == 'uploaded'
        assert names | {'assets-manifest.json', 'SHA256SUMS'} == set(remote)
        for name in remote:
            assert remote[name]['size'] == expected[name]['bytes']
            if remote[name].get('digest'):
                assert remote[name]['digest'] == 'sha256:' + expected[name]['sha256'], name
        gh('release', 'edit', tag, '--repo', repository, '--draft=false', '--latest')
        print('PUBLISHED', f'https://github.com/{repository}/releases/tag/{tag}', flush=True)


if __name__ == '__main__':
    main()
