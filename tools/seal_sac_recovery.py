"""Copy an atomic SAC recovery generation without changing its live producer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def seal(source, output):
    if output.exists():
        receipt = read(output/'SEAL.json')
        for name, value in receipt['files'].items():
            if sha(output/name) != value:
                raise ValueError('SEALED_FILE_CHANGED:' + name)
        return receipt
    temp = output.with_name(output.name+'.building')
    for attempt in range(3):
        if temp.exists():
            shutil.rmtree(temp)
        try:
            pointer = read(source/'RESUME.json')
            generation = pointer['directory']
            if Path(generation).name != generation:
                raise ValueError('RECOVERY_PATH')
            origin = source/'recovery'/generation
            if sha(origin/'MANIFEST.json') != pointer['manifest_sha256']:
                raise ValueError('RECOVERY_MANIFEST')
            temp.mkdir(parents=True)
            target = temp/'recovery'/generation
            shutil.copytree(origin, target)
            manifest = read(target/'MANIFEST.json')
            for name, value in manifest['files'].items():
                if Path(name).name != name or sha(target/name) != value:
                    raise ValueError('RECOVERY_FILE:' + name)
            state = read(target/'STATE.json')
            if state['identity'] != manifest['identity']:
                raise ValueError('RECOVERY_IDENTITY')
            for name in ('initial_bundle', 'bundles'):
                shutil.copytree(source/name, temp/name)
            for name in ('DESIGN.json', 'EFFECTIVE.json', 'EPOCH_ZERO.json'):
                shutil.copy2(source/name, temp/name)
            (temp/'RESUME.json').write_text(json.dumps(pointer, indent=2)+'\n')
            (temp/'HISTORY.json').write_text(json.dumps(state['history'], indent=2)+'\n')
            receipt = dict(schema='sac_recovery_seal.v1', identity=state['identity'],
                epoch=state['epoch'], steps=state['steps'], updates=state['updates'],
                generation=generation, files={str(p.relative_to(temp)): sha(p)
                    for p in sorted(temp.rglob('*')) if p.is_file()}, live_producer_modified=False)
            (temp/'SEAL.json').write_text(json.dumps(receipt, indent=2)+'\n')
            temp.rename(output)
            return receipt
        except FileNotFoundError:
            if attempt == 2:
                raise
    raise RuntimeError('UNREACHABLE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = seal(args.source, args.output)
    print(json.dumps({k: result[k] for k in ('epoch', 'steps', 'updates', 'identity')}), flush=True)
