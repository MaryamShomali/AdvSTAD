"""Four-model endpoint, persistence, and synthetic runner integration checks.

SMD is only an identifier in these tests. Every observation, label, provenance
record, POT result, and plot comes from a small in-memory fixture.
"""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import main
from src.experiment_tracking import ExperimentTracker


MODELS = ('TranAD', 'TranAD_Corrected', 'AdvSTAD_NoSpatial', 'AdvSTAD')
ADV_MODELS = ('AdvSTAD_NoSpatial', 'AdvSTAD')


def _alignment(name):
    return 'legacy_previous_endpoint' if name == 'TranAD' else 'inclusive_endpoint_v1'


def _configure(monkeypatch, tmp_path):
    torch.manual_seed(1)
    config = {
        'device': 'cpu',
        'dtype': 'float32',
        'advstad': {
            'window_size': 10,
            'batch_size': 3,
            'dropout': 0.0,
            'fusion': 'sum',
            'training': {'epochs_per_run': 2},
        },
        'experiment_tracking': {
            'enabled': True,
            'log_dir': str(tmp_path / 'runs'),
            'index_file': str(tmp_path / 'experiments.jsonl'),
        },
    }
    arguments = SimpleNamespace(
        model='TranAD', dataset='SMD', test=False, retrain=True,
        less=False, run_name='synthetic-ablation',
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, 'CONFIG', config)
    monkeypatch.setattr(main, 'args', arguments)
    monkeypatch.setattr(main, 'DEVICE', torch.device('cpu'))
    monkeypatch.setattr(main, 'DTYPE', torch.float32)
    return config, arguments


@pytest.mark.parametrize('rows,window', [(6, 3), (2, 5), (4, 1), (1, 1)])
def test_four_model_windows_preserve_legacy_and_include_only_observed_endpoints(rows, window):
    data = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    original = data.clone()
    converted = {}
    for name in MODELS:
        windows = main.convert_to_windows(data, SimpleNamespace(name=name, n_window=window))
        converted[name] = windows
        assert windows.shape == (rows, window, 3)
        for i in range(rows):
            endpoint = i - 1 if name == 'TranAD' else i
            indices = [max(0, j) for j in range(endpoint - window + 1, endpoint + 1)]
            assert torch.equal(windows[i], data[indices])
            assert max(indices) <= i
    assert torch.equal(converted['AdvSTAD_NoSpatial'], converted['AdvSTAD'])
    assert torch.equal(converted['TranAD_Corrected'], converted['AdvSTAD'])
    assert torch.equal(data, original)


@pytest.mark.parametrize('name', MODELS)
def test_scoring_reconstructs_the_selected_endpoint(monkeypatch, tmp_path, name):
    _, arguments = _configure(monkeypatch, tmp_path)
    arguments.model = name
    model, optimizer, scheduler, _, _ = main.load_model(name, 3)
    data = torch.linspace(0.05, 0.95, 15).reshape(5, 3)
    windows = main.convert_to_windows(data, model)
    model.eval()
    with torch.no_grad():
        scores, predictions = main.backprop(
            0, model, windows, 3, optimizer, scheduler, training=False,
        )
    target = torch.cat((data[:1], data[:-1])) if name == 'TranAD' else data
    assert scores.shape == predictions.shape == (5, 3)
    np.testing.assert_allclose(scores, (predictions - target.numpy()) ** 2, atol=1e-7)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_all_models_select_train_and_resume_their_own_checkpoints(monkeypatch, tmp_path):
    _, arguments = _configure(monkeypatch, tmp_path)
    data = torch.linspace(0.1, 0.9, 15).reshape(5, 3)
    saved = {}
    for name in MODELS:
        arguments.model = name
        model, optimizer, scheduler, epoch, history = main.load_model(name, 3)
        assert model.name == name and epoch == -1 and history == []
        windows = main.convert_to_windows(data, model)
        loss, learning_rate = main.backprop(0, model, windows, 3, optimizer, scheduler)
        assert np.isfinite(loss)
        history.append((loss, learning_rate))
        path = Path(main.save_model(model, optimizer, scheduler, 0, history))
        saved[name] = {
            'path': path,
            'bytes': path.read_bytes(),
            'state': {key: value.detach().clone() for key, value in model.state_dict().items()},
            'history': history,
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
    assert len({entry['path'] for entry in saved.values()}) == 4
    assert saved['TranAD']['path'] == Path('checkpoints/TranAD_SMD/model.ckpt')
    assert saved['TranAD_Corrected']['path'] == Path('checkpoints/TranAD_Corrected_SMD/model.ckpt')

    arguments.retrain = False
    for name in MODELS:
        arguments.model = name
        restored, optimizer, scheduler, epoch, history = main.load_model(name, 3)
        entry = saved[name]
        assert restored.name == name and epoch == 0 and history == entry['history']
        assert Path(main.model_experiment_key(restored)).name == entry['path'].parent.name
        assert entry['path'].read_bytes() == entry['bytes']
        for key, value in restored.state_dict().items():
            assert torch.equal(value, entry['state'][key])
        if name in ADV_MODELS:
            assert set(optimizer) == set(scheduler) == {'generator', 'adversary'}
            assert len(restored.training_history) == 1
            for owner in optimizer:
                assert optimizer[owner].state
                assert scheduler[owner].last_epoch == entry['scheduler'][owner].last_epoch
        else:
            assert optimizer.state
            assert scheduler.last_epoch == entry['scheduler'].last_epoch


def test_corrected_checkpoint_rejects_legacy_metadata_before_loading_weights(monkeypatch, tmp_path):
    _, arguments = _configure(monkeypatch, tmp_path)
    legacy, optimizer, scheduler, _, _ = main.load_model('TranAD', 3)
    with torch.no_grad():
        next(legacy.parameters()).fill_(0.123)
    source = Path(main.save_model(legacy, optimizer, scheduler, 0, []))
    arguments.model = 'TranAD_Corrected'
    corrected, _, _, _, _ = main.load_model(arguments.model, 3)
    before = {key: value.detach().clone() for key, value in corrected.state_dict().items()}
    target = Path('checkpoints') / main.model_experiment_key(corrected) / 'model.ckpt'
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    models = importlib.import_module('src.models')
    monkeypatch.setattr(models, 'TranAD_Corrected', lambda dims: corrected)
    arguments.retrain = False
    with pytest.raises(ValueError, match='Incompatible TranAD_Corrected checkpoint'):
        main.load_model(arguments.model, 3)
    for key, value in corrected.state_dict().items():
        assert torch.equal(value, before[key])


@pytest.mark.parametrize('resource_failure', [False, True])
def test_runner_failure_preserves_a_failed_record_and_error(monkeypatch, tmp_path, resource_failure):
    config, arguments = _configure(monkeypatch, tmp_path)
    captured = []

    def tracker_factory(*args, **kwargs):
        tracker = ExperimentTracker(*args, **kwargs)
        captured.append(tracker)
        return tracker

    def fail(tracker):
        raise RuntimeError('synthetic runner failure')

    monkeypatch.setattr(main, 'ExperimentTracker', tracker_factory)
    monkeypatch.setattr(main, 'run_experiment', fail)
    if resource_failure:
        def unavailable_resources(self, device):
            raise RuntimeError('synthetic CUDA resource failure')
        monkeypatch.setattr(ExperimentTracker, 'log_resource_usage', unavailable_resources)
    with pytest.raises(RuntimeError, match='synthetic runner failure'):
        main.main()
    assert len(captured) == 1
    tracker = captured[0]
    record = json.loads(tracker.summary_path.read_text())
    assert record['status'] == 'failed'
    assert record['model'] == arguments.model
    assert record['error'] == {'type': 'RuntimeError', 'message': 'synthetic runner failure'}
    assert record['timestamps']['completed_at'] is not None
    assert tracker.closed
    entries = [json.loads(line) for line in Path(config['experiment_tracking']['index_file']).read_text().splitlines()]
    assert len(entries) == 1 and entries[0]['run_id'] == record['run_id']
    assert entries[0]['status'] == 'failed'


def test_synthetic_runner_tracks_four_models_without_real_smd_io(monkeypatch, tmp_path):
    config, arguments = _configure(monkeypatch, tmp_path)
    train = torch.linspace(0.1, 0.8, 18).reshape(6, 3)
    test = torch.linspace(-0.1, 1.2, 15).reshape(5, 3)
    labels = np.zeros((5, 3), dtype=np.int64)
    labels[2, 1] = 1
    monkeypatch.setattr(
        main, 'load_dataset', lambda dataset: (train.clone(), test.clone(), labels.copy()),
    )
    monkeypatch.setattr(
        main, 'load_dataset_metadata',
        lambda dataset, feats: {'feature_names': ['sensor-0', 'sensor-1', 'sensor-2']},
    )
    monkeypatch.setattr(main, 'dataset_provenance', lambda dataset: {
        'machine_id': 'machine-1-1',
        'source_files': {'train': 'synthetic/train.npy', 'test': 'synthetic/test.npy'},
        'data_fingerprints': {'train': 'fake-train', 'test': 'fake-test', 'labels': 'fake-labels'},
    })
    clock = [0.0]
    events = []
    plotted = {}

    def timestamp(device):
        events.append(('timer', float(clock[0])))
        return clock[0]

    def fake_backprop(epoch, model, data, feats, optimizer, scheduler, training=True):
        if training:
            events.append(('train', model.name))
            clock[0] += 3.0
            model.last_training_metrics = {'loss_g': 0.25, 'lr_g': model.lr, 'alpha': 0.0}
            return 0.25, model.lr
        assert not model.training and not torch.is_grad_enabled()
        is_test = len(data) == len(test)
        events.append(('test-score' if is_test else 'calibration', model.name))
        clock[0] += 7.0 if is_test else 1000.0
        targets = data[:, -1].detach().cpu().numpy()
        return targets ** 2, np.zeros_like(targets)

    original_save = main.save_model

    def save_with_overhead(*args, **kwargs):
        clock[0] += 100.0
        return original_save(*args, **kwargs)

    def fake_pot(reference, scores, target_labels):
        clock[0] += 200.0
        return {
            'precision': np.float64(0.75), 'recall': np.float64(0.5),
            'f1': np.float64(0.6), 'TP': np.int64(1),
        }, np.zeros_like(target_labels)

    def fake_plot(key, observations, prediction, scores, target_labels):
        clock[0] += 300.0
        plotted[arguments.model] = (
            observations.clone(), prediction.copy(), scores.copy(), target_labels.copy(),
        )

    def fake_training_plot(*args, **kwargs):
        clock[0] += 400.0

    monkeypatch.setattr(main, 'synchronized_time', timestamp)
    monkeypatch.setattr(main, 'backprop', fake_backprop)
    monkeypatch.setattr(main, 'save_model', save_with_overhead)
    monkeypatch.setattr(main, 'pot_eval', fake_pot)
    monkeypatch.setattr(main, 'plotter', fake_plot)
    monkeypatch.setattr(main, 'plot_accuracies', fake_training_plot)
    monkeypatch.setattr(main, 'hit_att', lambda scores, target_labels: {})
    monkeypatch.setattr(main, 'ndcg', lambda scores, target_labels: {})

    records = {}
    snapshots = {}
    for name in MODELS:
        arguments.model = name
        tracker = ExperimentTracker(config, arguments, main.DEVICE, main.DTYPE)
        try:
            main.run_experiment(tracker)
        finally:
            tracker.complete()
        record = json.loads(tracker.summary_path.read_text())
        records[name] = record
        snapshots[tracker.summary_path] = tracker.summary_path.read_bytes()
        epochs = 2 if name in ADV_MODELS else 5
        assert record['model'] == name and record['dataset'] == 'SMD'
        assert record['seed'] == 1
        assert record['status'] == 'completed'
        assert record['dataset_details']['machine_id'] == 'machine-1-1'
        assert record['dataset_details']['window_alignment'] == _alignment(name)
        assert record['metrics']['evaluation']['overall']['f1'] == pytest.approx(0.6)
        assert record['metrics']['evaluation']['overall']['precision'] == pytest.approx(0.75)
        assert record['metrics']['evaluation']['overall']['recall'] == pytest.approx(0.5)
        runtime = record['metrics']['runtime']
        assert runtime['training_total_seconds'] == pytest.approx(3.0 * epochs)
        assert runtime['training_average_epoch_seconds'] == pytest.approx(3.0)
        assert runtime['training_epochs_this_run'] == epochs
        assert runtime['training_epochs_total'] == epochs
        assert runtime['inference_total_seconds'] == pytest.approx(7.0)
        assert runtime['inference_latency_ms_per_timepoint'] == pytest.approx(1400.0)
        assert record['hyperparameters']['trainable_parameter_count'] > 0
        assert record['checkpoint']['epoch'] == epochs - 1
        assert name in record['checkpoint']['path']
        assert record['environment']['device'] == 'cpu'
        assert record['environment']['dtype'] == 'float32'
        assert record['timestamps']['completed_at'] is not None
        observations, prediction, scores, plotted_labels = plotted[name]
        expected_observations = torch.roll(test, 1, 0) if name == 'TranAD' else test
        assert torch.equal(observations, expected_observations)
        np.testing.assert_array_equal(plotted_labels, labels)
        target = torch.cat((test[:1], test[:-1])) if name == 'TranAD' else test
        np.testing.assert_allclose(scores, (prediction - target.numpy()) ** 2, atol=1e-7)

    assert len({record['run_id'] for record in records.values()}) == 4
    assert len({record['checkpoint']['path'] for record in records.values()}) == 4
    assert len(list((tmp_path / 'runs').glob('*/run.json'))) == 4
    # The only narrowly timed operations are training compute and test scoring.
    for index, (event, _) in enumerate(events):
        if event in ('train', 'test-score'):
            assert events[index - 1][0] == events[index + 1][0] == 'timer'

    arguments.test = True
    for name in MODELS:
        arguments.model = name
        tracker = ExperimentTracker(config, arguments, main.DEVICE, main.DTYPE)
        try:
            main.run_experiment(tracker)
        finally:
            tracker.complete()
        record = json.loads(tracker.summary_path.read_text())
        runtime = record['metrics']['runtime']
        assert record['checkpoint']['loaded'] is True
        assert record['checkpoint']['path'] == records[name]['checkpoint']['path']
        assert record['metrics']['training'] == []
        assert runtime['training_epochs_this_run'] == 0
        assert runtime['training_epochs_total'] == records[name]['metrics']['runtime']['training_epochs_total']
        assert runtime['training_total_seconds'] is None
        assert runtime['training_average_epoch_seconds'] is None
        assert runtime['inference_total_seconds'] == pytest.approx(7.0)
        assert runtime['inference_latency_ms_per_timepoint'] == pytest.approx(1400.0)
    for path, content in snapshots.items():
        assert path.read_bytes() == content
    index_records = [json.loads(line) for line in (tmp_path / 'experiments.jsonl').read_text().splitlines()]
    assert len(index_records) == 8
    assert len({record['run_id'] for record in index_records}) == 8


@pytest.mark.parametrize('device_type', ['cpu', 'cuda'])
def test_runtime_timer_synchronizes_cuda_before_reading_monotonic_clock(monkeypatch, device_type):
    runtime = importlib.import_module('src.runtime_measurement')
    events = []
    device = torch.device(device_type)

    def synchronize(actual_device):
        assert actual_device == device
        events.append('synchronize')

    def perf_counter():
        events.append('clock')
        return 123.5

    monkeypatch.setattr(runtime.torch.cuda, 'synchronize', synchronize)
    monkeypatch.setattr(runtime, 'perf_counter', perf_counter)
    assert runtime.synchronized_time(device) == 123.5
    assert events == (['synchronize', 'clock'] if device_type == 'cuda' else ['clock'])
