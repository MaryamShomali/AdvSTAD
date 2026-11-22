"""Checkpoint/record protection using temporary files and synthetic metadata."""

from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import main
import src.experiment_tracking as tracking
from src.experiment_tracking import ExperimentTracker
from src.reproducibility import seed_everything
from src.study_protocol import ABLATION_MODELS


def configure(monkeypatch, tmp_path, model='TranAD'):
    seed_everything(1)
    config = {
        'device': 'cpu', 'dtype': 'float32', 'seed': 1,
        'advstad': {'fusion': 'concat'},
        'experiment_tracking': {'enabled': True, 'log_dir': str(tmp_path / 'runs'),
                                'index_file': str(tmp_path / 'experiments.jsonl')},
    }
    args = SimpleNamespace(model=model, dataset='synthetic', test=False,
                           retrain=True, less=False, run_name='fixture')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, 'CONFIG', config)
    monkeypatch.setattr(main, 'args', args)
    monkeypatch.setattr(main, 'DEVICE', torch.device('cpu'))
    monkeypatch.setattr(main, 'DTYPE', torch.float32)
    return config, args


@pytest.mark.parametrize('model', ABLATION_MODELS)
@pytest.mark.parametrize('retrain', [False, True])
def test_test_only_missing_checkpoint_never_returns_a_random_model(monkeypatch, tmp_path, model, retrain):
    _, args = configure(monkeypatch, tmp_path, model)
    args.test, args.retrain = True, retrain
    with pytest.raises(FileNotFoundError, match='--test requires an existing trained checkpoint'):
        main.load_model(model, 3)
    assert not (tmp_path / 'checkpoints').exists()


@pytest.mark.parametrize('model', ABLATION_MODELS)
def test_test_only_rejects_untrained_checkpoint_without_changing_it(monkeypatch, tmp_path, model):
    _, args = configure(monkeypatch, tmp_path, model)
    instance, optimizer, scheduler, _, history = main.load_model(model, 3)
    path = Path(main.save_model(instance, optimizer, scheduler, -1, history))
    before = path.read_bytes()
    args.test = True
    with pytest.raises(ValueError, match='at least one completed epoch'):
        main.load_model(model, 3)
    assert path.read_bytes() == before


@pytest.mark.parametrize('model', ABLATION_MODELS)
def test_test_only_reports_corrupt_checkpoint_path(monkeypatch, tmp_path, model):
    _, args = configure(monkeypatch, tmp_path, model)
    instance, _, _, _, _ = main.load_model(model, 3)
    path = Path('checkpoints') / main.model_experiment_key(instance) / 'model.ckpt'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'not a checkpoint')
    args.test = True
    with pytest.raises(ValueError, match='Could not load the trained checkpoint required by --test') as error:
        main.load_model(model, 3)
    assert str(path) in str(error.value)
    assert path.read_bytes() == b'not a checkpoint'


@pytest.mark.parametrize('model', ABLATION_MODELS)
def test_test_only_rejects_incomplete_checkpoint_before_loading_weights(monkeypatch, tmp_path, model):
    _, args = configure(monkeypatch, tmp_path, model)
    instance, _, _, _, _ = main.load_model(model, 3)
    path = Path('checkpoints') / main.model_experiment_key(instance) / 'model.ckpt'
    path.parent.mkdir(parents=True)
    torch.save({'epoch': 0, 'model_state_dict': instance.state_dict()}, path)
    before = path.read_bytes()
    args.test = True
    with pytest.raises(ValueError, match='Invalid trained checkpoint required by --test') as error:
        main.load_model(model, 3)
    assert str(path) in str(error.value)
    assert path.read_bytes() == before


@pytest.mark.parametrize('model', ABLATION_MODELS)
def test_previous_checkpoint_is_archived_once_without_changing_load_paths(monkeypatch, tmp_path, model):
    _, args = configure(monkeypatch, tmp_path, model)
    original, opt, sched, _, _ = main.load_model(model, 3)
    path = Path(main.save_model(original, opt, sched, 0, []))
    before = path.read_bytes()
    replacement, opt, sched, _, _ = main.load_model(model, 3)
    with torch.no_grad():
        next(replacement.parameters()).fill_(0.123)
    assert Path(main.save_model(replacement, opt, sched, 0, [])) == path
    archive = Path(replacement.previous_checkpoint_archive)
    assert archive.read_bytes() == before
    assert path.read_bytes() != before
    main.save_model(replacement, opt, sched, 1, [])
    assert list(archive.parent.glob('*.ckpt')) == [archive]
    assert archive.read_bytes() == before
    # The schema and normal resume path stay compatible with existing loaders.
    args.test = True
    loaded, _, _, epoch, _ = main.load_model(model, 3)
    assert epoch == 1
    for key, value in replacement.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


@pytest.mark.parametrize('model', ABLATION_MODELS)
def test_actual_architecture_and_rng_settings_are_serialized_for_all_models(monkeypatch, tmp_path, model):
    config, args = configure(monkeypatch, tmp_path, model)
    instance, opt, sched, _, _ = main.load_model(model, 3)
    tracker = ExperimentTracker(config, args, main.DEVICE, main.DTYPE)
    tracker.log_dataset(torch.zeros(5, 3), torch.zeros(4, 3), np.zeros((4, 3)))
    tracker.log_model(instance, opt, sched, 0, True, 5)
    tracker.log_runtime(training_total_seconds=2.0, training_average_epoch_seconds=.4,
                        training_epochs_this_run=5, training_epochs_total=5,
                        inference_total_seconds=.1, inference_latency_ms_per_timepoint=25.)
    tracker.log_evaluation(np.zeros((5, 3)), np.zeros((4, 3)),
                           {'precision': .8, 'recall': .6, 'f1': .6857}, pd.DataFrame([{'f1': .1}]))
    tracker.complete()
    record = json.loads(tracker.summary_path.read_text())
    actual = record['hyperparameters']['effective_config']
    assert actual['model'] == model
    assert actual['d_model'] == 6 and actual['nhead'] == 3
    assert actual['dim_feedforward'] == 16 and actual['dropout'] == .1
    assert actual['attention_dropout'] == .1
    assert actual['training']['epochs_per_run'] == 5
    assert actual['training']['learning_rate'] == instance.lr
    assert actual['seed'] == record['seed'] == 1
    assert record['reproducibility']['python_seed'] == record['reproducibility']['numpy_seed'] == 1
    assert record['metrics']['evaluation']['overall']['precision'] == .8
    assert record['metrics']['runtime']['inference_total_seconds'] == .1
    assert record['metrics']['runtime']['training_total_seconds'] == 2.
    if model.startswith('AdvSTAD'):
        assert actual['output_heads'] == 'separate'
        assert actual['training']['optimizer_updates_per_batch'] == 2
        assert actual['fusion'] == ('concat' if model == 'AdvSTAD' else 'temporal_only')
        assert actual['spatial_layers'] == (1 if model == 'AdvSTAD' else 0)
    else:
        assert actual['fusion'] is None and actual['output_heads'] == 'shared'
        assert actual['training']['gradient_clip_norm'] is None
        assert actual['training']['optimizer_updates_per_batch'] == 1


def test_run_id_collision_cannot_overwrite_an_existing_record(monkeypatch, tmp_path):
    config, args = configure(monkeypatch, tmp_path)

    class FrozenDatetime:
        @staticmethod
        def now():
            return datetime.fromisoformat('2026-11-22T12:00:00+03:30')

    monkeypatch.setattr(tracking, 'datetime', FrozenDatetime)
    first = ExperimentTracker(config, args, main.DEVICE, main.DTYPE)
    first.complete()
    before = first.summary_path.read_bytes()
    second = ExperimentTracker(config, args, main.DEVICE, main.DTYPE)
    second.complete()
    assert first.run_dir != second.run_dir
    assert first.summary_path.read_bytes() == before


@pytest.mark.parametrize('model', ABLATION_MODELS)
def test_seed_provenance_rejects_mismatch_but_accepts_legacy_payloads(monkeypatch, tmp_path, model):
    _, args = configure(monkeypatch, tmp_path, model)
    instance, opt, sched, _, _ = main.load_model(model, 3)
    path = Path(main.save_model(instance, opt, sched, 0, []))
    before = path.read_bytes()
    seed_everything(2)
    args.test = True
    with pytest.raises(ValueError, match='Checkpoint training seed 1 differs from the selected seed 2'):
        main.load_model(model, 3)
    assert path.read_bytes() == before
    # Optional fields do not invalidate existing pre-study checkpoint formats.
    legacy = torch.load(path, map_location='cpu')
    legacy.pop('training_seed')
    torch.save(legacy, path)
    loaded, _, _, epoch, _ = main.load_model(model, 3)
    assert epoch == 0 and loaded.training_seed is None
    seed_everything(1)


def test_repeated_synthetic_runs_keep_distinct_immutable_plots(monkeypatch, tmp_path):
    config, args = configure(monkeypatch, tmp_path)
    train = torch.zeros(5, 3)
    test = torch.zeros(4, 3)
    labels = np.zeros((4, 3))
    monkeypatch.setattr(main, 'load_dataset', lambda dataset: (train.clone(), test.clone(), labels.copy()))
    monkeypatch.setattr(main, 'load_dataset_metadata', lambda *args: {})
    monkeypatch.setattr(main, 'backprop', lambda epoch, model, data, feats, opt, sched, training=True:
                        (.1, model.lr) if training else (np.zeros((len(data), 3)), np.zeros((len(data), 3))))
    monkeypatch.setattr(main, 'pot_eval', lambda reference, score, target: ({'f1': .1}, np.zeros_like(target)))
    monkeypatch.setattr(main, 'hit_att', lambda *args: {})
    monkeypatch.setattr(main, 'ndcg', lambda *args: {})

    def plot(key, filename):
        path = Path('plots') / key / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key)

    monkeypatch.setattr(main, 'plot_accuracies', lambda values, key: plot(key, 'training-graph.pdf'))
    monkeypatch.setattr(main, 'plotter', lambda key, *args: plot(key, 'output.pdf'))
    snapshots = {}
    paths = []
    for _ in range(2):
        tracker = ExperimentTracker(config, args, main.DEVICE, main.DTYPE)
        main.run_experiment(tracker)
        tracker.complete()
        record = json.loads(tracker.summary_path.read_text())
        for name in ('training_plot', 'evaluation_plot'):
            path = Path(record['artifacts'][name])
            assert record['run_id'] in str(path)
            paths.append(path)
            snapshots[path] = path.read_bytes()
    assert len(set(paths)) == 4
    assert all(path.read_bytes() == data for path, data in snapshots.items())
