"""Epoch stopping and self-contained inference/recovery artifacts for discrete SAC."""
import copy
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from agent_plugins.three_signal_sac import ThreeSignalSAC
from app.frozen_recipe_inputs import file_sha, validate_recipe
from app.rl_subset import digest
from tools.run_discrete_curve_probe import write


class EpochPatience:
    MODES = ('mean', 'validation', 'mean_no_regression')

    def __init__(self, patience=20, min_delta=1e-8, mode='mean', start_epoch=1,
                 select_during_warmup=False):
        if (type(patience) is not int or patience < 1 or not np.isfinite(min_delta)
                or min_delta < 0 or mode not in self.MODES
                or type(start_epoch) is not int or start_epoch < 1):
            raise ValueError('EPOCH_STOPPING_CONFIG')
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.start_epoch, self.epoch = start_epoch, 0
        self.select_during_warmup = select_during_warmup
        self.best, self.bad = None, 0
        self.previous_train = self.previous_validation = self.previous_mean = None
        self.best_train = self.best_validation = None

    def observe(self, train, validation):
        if not np.isfinite([train, validation]).all():
            raise ValueError('EPOCH_MONITOR_NOT_FINITE')
        train, validation = float(train), float(validation)
        mean = .5 * train + .5 * validation
        score = validation if self.mode == 'validation' else mean
        self.epoch += 1
        previous = dict(previous_train=self.previous_train,
                        previous_validation=self.previous_validation,previous_mean=self.previous_mean)
        if self.epoch < self.start_epoch and not self.select_during_warmup:
            return dict(score=score,mean=mean,mode=self.mode,improved=False,best=None,
                        best_train=None,best_validation=None,bad=0,stop=False,
                        epoch=self.epoch,warming_up=True,rejections=['MONITOR_WARMUP'],**previous)
        rejections = []
        if self.mode == 'mean_no_regression' and self.previous_mean is not None:
            for value, old, name in ((train,self.previous_train,'TRAIN'),
                                     (validation,self.previous_validation,'VALIDATION'),
                                     (mean,self.previous_mean,'MEAN')):
                if value < old:
                    rejections.append(name+'_DECREASE')
        if self.best is not None and score <= self.best + self.min_delta:
            rejections.append('NO_NEW_BEST')
        improved = not rejections
        if improved:
            self.best, self.bad = score, 0
            self.best_train, self.best_validation = train, validation
        elif not (self.select_during_warmup and self.epoch <= self.start_epoch):
            self.bad += 1
        self.previous_train, self.previous_validation, self.previous_mean = train, validation, mean
        return dict(score=score, mean=mean, mode=self.mode, improved=improved, best=self.best,
                    best_train=self.best_train,best_validation=self.best_validation,bad=self.bad,
                    stop=self.bad >= self.patience,rejections=rejections,
                    epoch=self.epoch,warming_up=self.select_during_warmup and self.epoch <= self.start_epoch,**previous)


def capture_rng(rng):
    return dict(generator=copy.deepcopy(rng.bit_generator.state), numpy=np.random.get_state(),
                python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state, rng):
    rng.bit_generator.state = state['generator']
    np.random.set_state(state['numpy']); random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def build_model(recipe, effective, seed=7, device='cpu'):
    return ThreeSignalSAC(recipe, device=device, seed=seed, gamma=effective['gamma'],
        alpha=effective['alpha'], lr=effective['learning_rate'],
        entropy_fraction=effective['entropy_fraction'], architecture=effective['architecture'],
        tau=effective['tau'], weight_decay=effective['weight_decay'],
        critic_regularization=effective.get('critic_regularization'))


def durable_files(directory):
    for path in Path(directory).iterdir():
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_files(directory, files):
    directory = Path(directory).resolve()
    for name, expected in files.items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or file_sha(path) != expected:
            raise ValueError('BUNDLE_HASH:' + name)


def export_bundle(path, model, probe, checkpoint):
    """The inference bundle has no dependency on replay, training data or a refit."""
    path = Path(path)
    if path.exists():
        raise ValueError('BUNDLE_ALREADY_EXISTS')
    temp = path.with_name(path.name + '.building')
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    write(temp/'RECIPE.json', model.recipe)
    write(temp/'MODEL_CONFIG.json', model.effective)
    model.save(temp/'model.pt')
    with torch.inference_mode():
        logits = model.surfaces(torch.tensor(probe, device=model.device)).cpu().numpy()
    np.savez_compressed(temp/'PROBE.npz', inputs=probe, logits=logits)
    manifest = dict(schema='structural_sac.inference.v1', checkpoint=checkpoint,
        recipe_identity=model.recipe['identity'], columns=model.recipe['list']['columns'],
        codec='NOP_BUY_SELL_CLOSE_OPPOSITE_V1',
        backend=dict(torch=torch.__version__, device=str(model.device),
                     cudnn_tf32=torch.backends.cudnn.allow_tf32),
        files={p.name:file_sha(p) for p in temp.iterdir() if p.is_file()})
    write(temp/'MANIFEST.json', manifest)
    # Cold construction must not perturb the continuing training RNG stream.
    with torch.random.fork_rng():
        restored, _ = load_bundle(temp, str(model.device))
        with torch.inference_mode():
            replay = restored.surfaces(torch.tensor(probe, device=model.device)).cpu().numpy()
        if not np.array_equal(logits, replay):
            raise ValueError('BUNDLE_RELOAD_NOT_EXACT')
    write(temp/'RELOAD.json', dict(exact=True, rows=len(probe), max_abs_error=0.,
                                 checkpoint_sha256=file_sha(temp/'model.pt')))
    durable_files(temp)
    temp.rename(path)
    durable_files(path.parent)
    return dict(path=path.name, manifest_sha256=file_sha(path/'MANIFEST.json'),
                model_sha256=file_sha(path/'model.pt'), reload_exact=True)


def load_bundle(path, device='cpu'):
    path = Path(path)
    manifest = json.loads((path/'MANIFEST.json').read_text())
    if manifest['schema'] != 'structural_sac.inference.v1':
        raise ValueError('BUNDLE_SCHEMA')
    required = {'RECIPE.json', 'MODEL_CONFIG.json', 'model.pt', 'PROBE.npz'}
    if set(manifest['files']) != required:
        raise ValueError('BUNDLE_FILES')
    verify_files(path, manifest['files'])
    recipe = json.loads((path/'RECIPE.json').read_text())
    validate_recipe(recipe)
    if recipe['identity'] != manifest['recipe_identity'] or recipe['list']['columns'] != manifest['columns']:
        raise ValueError('BUNDLE_RECIPE')
    effective = json.loads((path/'MODEL_CONFIG.json').read_text())
    model = build_model(recipe, effective, device=device)
    model.load(path/'model.pt')
    return model, manifest


def save_recovery(output, model, replay, rng, state):
    """Commit one complete epoch; retain the previous generation for recovery."""
    output = Path(output)
    root = output/'recovery'; root.mkdir(exist_ok=True)
    name = 'epoch_%04d' % state['epoch']
    final, temp = root/name, root/(name+'.building')
    if temp.exists():
        shutil.rmtree(temp)
    if final.exists():
        # Only an uncommitted duplicate of this epoch can be replaced by the caller.
        raise ValueError('RECOVERY_EPOCH_ALREADY_EXISTS')
    temp.mkdir()
    model.save(temp/'model.pt')
    torch.save(dict(replay=replay.state_dict(), rng=capture_rng(rng)), temp/'learning.pt')
    write(temp/'STATE.json', state)
    files = {p.name:file_sha(p) for p in temp.iterdir() if p.is_file()}
    write(temp/'MANIFEST.json', dict(identity=state['identity'], files=files))
    durable_files(temp); temp.rename(final); durable_files(root)
    write(output/'RESUME.json', dict(directory=name, manifest_sha256=file_sha(final/'MANIFEST.json')))
    durable_files(output)
    for old in sorted(p for p in root.glob('epoch_*') if p.is_dir() and not p.name.endswith('.building'))[:-2]:
        shutil.rmtree(old)


def load_recovery(output, model, replay, rng, identity):
    output = Path(output)
    pointer = json.loads((output/'RESUME.json').read_text())
    folder = output/'recovery'/pointer['directory']
    if folder.parent.resolve() != (output/'recovery').resolve():
        raise ValueError('RECOVERY_PATH')
    if file_sha(folder/'MANIFEST.json') != pointer['manifest_sha256']:
        raise ValueError('RECOVERY_MANIFEST')
    manifest = json.loads((folder/'MANIFEST.json').read_text())
    if manifest['identity'] != identity:
        raise ValueError('RECOVERY_IDENTITY')
    verify_files(folder, manifest['files'])
    state = json.loads((folder/'STATE.json').read_text())
    if state['identity'] != identity:
        raise ValueError('RECOVERY_STATE_IDENTITY')
    model.load(folder/'model.pt')
    learning = torch.load(folder/'learning.pt', map_location='cpu', weights_only=False)
    replay.load_state_dict(learning['replay'])
    restore_rng(learning['rng'], rng)
    return state
