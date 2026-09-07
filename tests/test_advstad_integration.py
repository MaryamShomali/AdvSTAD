"""Timestamp, runner, tracking, and legacy persistence integration checks."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import main
from src.advstad_checkpoint import experiment_key
from src.experiment_tracking import ExperimentTracker
from src.models import AdvSTAD


@pytest.mark.parametrize('rows,window', [(6, 3), (2, 5), (4, 1), (1, 1)])
def test_inclusive_windows_and_legacy_endpoints(rows, window):
    data = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    original = data.clone()
    for name in ('AdvSTAD', 'TranAD'):
        windows = main.convert_to_windows(data, SimpleNamespace(name=name, n_window=window))
        assert tuple(windows.shape) == (rows, window, 3)
        for i in range(rows):
            endpoint = i if name == 'AdvSTAD' else i - 1
            indices = torch.tensor([max(0, j) for j in range(endpoint - window + 1, endpoint + 1)])
            assert torch.equal(windows[i], data[indices])
        # Rows overlap in storage without copying the full windowed dataset.
        assert windows.stride() == (3, 3, 1)
    assert torch.equal(data, original)


@pytest.mark.parametrize('window,rows', [(4, 7), (5, 2), (1, 5)])
def test_evaluation_scores_refer_to_current_timestamp(window, rows):
    model = AdvSTAD(3, {'window_size': window, 'batch_size': 2, 'dropout': 0.0})
    data = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3) / 30
    windows = main.convert_to_windows(data, model)
    scores, prediction = main.backprop(0, model, windows, 3, None, None, training=False)
    assert scores.shape == prediction.shape == tuple(data.shape)
    np.testing.assert_allclose(scores, (prediction - data.numpy()) ** 2, atol=1e-7)
    individual = []
    for i in range(rows):
        _, output = main.backprop(0, model, windows[i:i + 1], 3, None, None, training=False)
        individual.append(output)
    np.testing.assert_allclose(prediction, np.concatenate(individual), atol=1e-6)


@pytest.mark.parametrize('shapes', [((4, 2), (3, 3), (3, 3)), ((4, 2), (3, 2), (2, 2)),
                                   ((0, 2), (3, 2), (3, 2)), ((4,), (3, 2), (3, 2))])
def test_data_shape_disagreements_fail_before_construction(shapes):
    with pytest.raises(ValueError):
        main.validate_dataset_shapes(*(np.zeros(shape) for shape in shapes))


def configure_runner(monkeypatch, tmp_path, fusion='sum', model='AdvSTAD'):
    config = {
        'device': 'cpu', 'dtype': 'float32',
        'advstad': {'fusion': fusion, 'window_size': 4, 'batch_size': 3, 'dropout': 0.0,
                    'training': {'epochs_per_run': 2}},
        'experiment_tracking': {'enabled': True, 'log_dir': str(tmp_path / 'runs'),
                                'index_file': str(tmp_path / 'experiments.jsonl')},
    }
    arguments = SimpleNamespace(model=model, dataset='synthetic', test=False, retrain=True,
                                less=False, run_name='integration')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, 'args', arguments)
    monkeypatch.setattr(main, 'CONFIG', config)
    monkeypatch.setattr(main, 'DEVICE', torch.device('cpu'))
    monkeypatch.setattr(main, 'DTYPE', torch.float32)
    return config, arguments


@pytest.mark.parametrize('fusion', ['sum', 'concat', 'cross_attention'])
def test_runner_train_resume_test_only_and_tracking(monkeypatch, tmp_path, fusion):
    config, arguments = configure_runner(monkeypatch, tmp_path, fusion)
    train = torch.linspace(0.05, 0.85, 24).reshape(8, 3)
    test = torch.linspace(-0.1, 1.2, 21).reshape(7, 3)
    labels = np.zeros((7, 3))
    labels[3, 1] = 1
    monkeypatch.setattr(main, 'load_dataset', lambda dataset: (train.clone(), test.clone(), labels.copy()))
    # POT's tail fitting needs longer series; test its exact inputs here.
    pot_inputs = []
    def fake_pot(reference, score, target_labels):
        pot_inputs.append((reference.copy(), score.copy(), target_labels.copy()))
        return {'f1': 0.0}, np.zeros_like(target_labels)
    monkeypatch.setattr(main, 'pot_eval', fake_pot)
    plotted = []
    def fake_plot(key, observations, prediction, scores, target_labels):
        plotted.append((key, observations.clone(), prediction.copy(), scores.copy(), target_labels.copy()))
        path = Path('plots') / key / 'output.pdf'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'plot fixture')
    def fake_training_plot(history, key, loss_label=None):
        assert loss_label == 'Generator objective'
        assert len(history) == 2
        path = Path('plots') / key / 'training-graph.pdf'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'training plot fixture')
    monkeypatch.setattr(main, 'plotter', fake_plot)
    monkeypatch.setattr(main, 'plot_accuracies', fake_training_plot)
    tracker = ExperimentTracker(config, arguments, main.DEVICE, main.DTYPE)
    try:
        main.run_experiment(tracker)
    finally:
        tracker.complete()
    record = json.loads(tracker.summary_path.read_text())
    key = record['experiment_key']
    assert key.startswith('AdvSTAD_synthetic_' + fusion + '_')
    assert set(record['hyperparameters']['optimizers']) == {'generator', 'adversary'}
    assert set(record['hyperparameters']['schedulers']) == {'generator', 'adversary'}
    assert record['hyperparameters']['advstad']['d_model'] == 6
    assert record['dataset_details']['window_alignment'] == 'inclusive_endpoint_v1'
    assert record['dataset_details']['observation_ranges']['test']['max'] > 1
    assert record['metrics']['training'][0]['alpha'] == 0
    assert record['metrics']['training'][1]['alpha'] == pytest.approx(0.05)
    for metrics in record['metrics']['training']:
        assert {'loss_g', 'loss_a', 'error_y1', 'error_y2_base', 'error_y2', 'lr_g', 'lr_a'} <= set(metrics)
    for artifact in ['training_plot', 'evaluation_plot']:
        assert key in record['artifacts'][artifact]
        assert Path(record['artifacts'][artifact]).exists()
    assert key in record['checkpoint']['path']
    assert torch.equal(plotted[0][1], test)
    np.testing.assert_array_equal(plotted[0][4], labels)
    np.testing.assert_allclose(plotted[0][3], (plotted[0][2] - test.numpy()) ** 2, atol=1e-7)

    arguments.test = True
    # --test still loads even when --retrain was also supplied, as in legacy CLI.
    tracker_test = ExperimentTracker(config, arguments, main.DEVICE, main.DTYPE)
    try:
        main.run_experiment(tracker_test)
    finally:
        tracker_test.complete()
    assert tracker_test.record['checkpoint']['loaded'] is True
    assert tracker_test.record['checkpoint']['epoch'] == 1
    assert tracker_test.record['experiment_key'] == key
    assert tracker_test.record['metrics']['training'] == []
    assert len(plotted) == 1
    for previous, loaded in zip(pot_inputs[:4], pot_inputs[4:]):
        for left, right in zip(previous, loaded):
            np.testing.assert_allclose(left, right, atol=1e-7)
    restored, optimizers, schedulers, epoch, history = main.load_model('AdvSTAD', 3)
    assert epoch == 1 and len(history) == len(restored.training_history) == 2
    assert experiment_key(restored, arguments.dataset) == key
    windows = main.convert_to_windows(train, restored)
    main.backprop(epoch + 1, restored, windows, 3, optimizers, schedulers)
    assert restored.last_training_metrics['alpha'] == pytest.approx(0.1 * (1 - 1 / 3))


def test_atomic_checkpoint_failure_preserves_previous_save(monkeypatch, tmp_path):
    configure_runner(monkeypatch, tmp_path)
    model, optimizers, schedulers, _, history = main.load_model('AdvSTAD', 3)
    path = Path(main.save_model(model, optimizers, schedulers, -1, history))
    saved = path.read_bytes()
    def fail_save(payload, temporary_path):
        Path(temporary_path).write_bytes(b'incomplete')
        raise OSError('simulated interrupted write')
    monkeypatch.setattr(main.torch, 'save', fail_save)
    with pytest.raises(OSError, match='interrupted'):
        main.save_model(model, optimizers, schedulers, 0, history)
    assert path.read_bytes() == saved
    assert not Path(str(path) + '.tmp').exists()


def test_legacy_checkpoint_and_tracker_calls_still_work(monkeypatch, tmp_path):
    config, arguments = configure_runner(monkeypatch, tmp_path, model='TranAD')
    # Unselected malformed AdvSTAD settings must not affect legacy loading.
    config['advstad'] = {'fusion': 'invalid'}
    model, optimizer, scheduler, _, _ = main.load_model('TranAD', 3)
    state = copy.deepcopy(model.state_dict())
    path = main.save_model(model, optimizer, scheduler, 2, [(0.2, model.lr)])
    assert Path(path) == Path('checkpoints/TranAD_synthetic/model.ckpt')
    arguments.retrain = False
    restored, restored_optimizer, restored_scheduler, epoch, history = main.load_model('TranAD', 3)
    assert epoch == 2 and history == [(0.2, model.lr)]
    for name, value in restored.state_dict().items():
        assert torch.equal(value, state[name])
    tracker = ExperimentTracker(config, arguments, main.DEVICE, main.DTYPE)
    try:
        tracker.log_model(restored, restored_optimizer, restored_scheduler, 3, True, 5)
        tracker.log_training_epoch(3, 0.1, model.lr)
    finally:
        tracker.complete()
    assert tracker.record['hyperparameters']['optimizer']['name'] == 'AdamW'
    assert tracker.record['dataset_details']['window_alignment'] == 'legacy_previous_endpoint'
    assert tracker.record['metrics']['training'][0]['loss'] == 0.1
