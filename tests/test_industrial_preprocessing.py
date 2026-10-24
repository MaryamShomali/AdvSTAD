"""Release-format, leakage, and label alignment checks using small raw files."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from openpyxl import Workbook

import main
import preprocess
from src.advstad_checkpoint import checkpoint_metadata, experiment_key
from src.experiment_tracking import ExperimentTracker
from src.industrial_preprocessing import WADI_LABEL, _prepare_features, preprocess_industrial


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        csv.writer(handle).writerows(rows)


def wadi_release(root, labels=(1, -1, 1)):
    raw = root / 'raw'
    write_csv(raw / 'WADI_14days_new.csv', [
        ['Row', 'Date', 'Time', 'SENSOR_A', 'SENSOR_B', '2_LS_001_AL'],
        [1, '9/25/2017', '00:00.0', 10, 7, ''],
        [2, '9/25/2017', '00:01.0', '', 7, ''],
        [3, '9/25/2017', '00:02.0', 20, 7, ''],
    ])
    # Test features are deliberately reordered; feature identity must win.
    write_csv(raw / 'WADI_attackdataLABLE.csv', [
        list(range(7)),
        ['Row ', 'Date ', 'Time', 'SENSOR_B', '2_LS_001_AL', 'SENSOR_A', WADI_LABEL],
        [1, '10/9/17', '00:00.0', 7, '', '', labels[0]],
        [2, '10/9/17', '00:01.0', 7, '', 30, labels[1]],
        [3, '10/9/17', '00:02.0', 7, '', '', labels[2]],
        ['', '', '', '', '', '', 1],
        ['', '', '', '', '', '', 1],
    ])


def write_swat(path, sheet_name, columns, observations):
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    sheet.append([None, 'P1'])
    sheet.append(columns)
    for row in observations:
        sheet.append(row)
    workbook.save(path)
    workbook.close()


def swat_release(root, attack_labels=('Normal', 'Attack', 'A ttack')):
    physical = root / 'raw' / 'Physical'
    columns = [' Timestamp', ' SENSOR_A ', 'SENSOR_B', 'Normal/Attack ']
    write_swat(physical / 'SWaT_Dataset_Normal_v1.xlsx', 'Normal.csv', columns, [
        ['22/12/2015 4:30:00 PM', 10, 7, 'Normal'],
        ['22/12/2015 4:30:01 PM', 20, 7, 'Normal'],
    ])
    write_swat(physical / 'SWaT_Dataset_Attack_v0.xlsx', 'Combined Data', columns, [
        ['28/12/2015 10:00:00 AM', 15, 7, attack_labels[0]],
        ['28/12/2015 10:00:01 AM', 30, 7, attack_labels[1]],
        ['28/12/2015 10:00:02 AM', None, 7, attack_labels[2]],
    ])


def arrays(folder):
    return [np.load(folder / f'{name}.npy') for name in ('train', 'test', 'labels')]


def test_wadi_new_release_scaling_missing_data_and_footer(tmp_path):
    wadi_release(tmp_path / 'WADI')
    destination = tmp_path / 'processed'
    metadata = preprocess_industrial('WADI', tmp_path / 'WADI', destination)
    train, test, labels = arrays(destination)
    np.testing.assert_allclose(train, [[0, 0], [0, 0], [1, 0]])
    # Initial gaps use the normal-data median, later gaps use past test values.
    # A test value above the normal range is retained above 1, not independently scaled.
    np.testing.assert_allclose(test, [[0.5, 0], [2, 0], [2, 0]])
    np.testing.assert_array_equal(labels, [[0, 0], [1, 1], [0, 0]])
    assert labels.dtype == np.uint8
    assert metadata['feature_names'] == ['SENSOR_A', 'SENSOR_B']
    assert metadata['dropped_features'] == ['2_LS_001_AL']
    assert metadata['source_rows'] == {'train': 3, 'test': 3}
    assert metadata['label_scope'] == 'timestamp'
    assert metadata['attack_timestamps'] == 1
    assert json.loads((destination / 'metadata.json').read_text()) == metadata


def test_swat_workbook_preamble_whitespace_and_attack_spelling(tmp_path):
    swat_release(tmp_path / 'SWaT')
    destination = tmp_path / 'processed'
    metadata = preprocess_industrial('SWaT', tmp_path / 'SWaT', destination)
    train, test, labels = arrays(destination)
    np.testing.assert_allclose(train, [[0, 0], [1, 0]])
    np.testing.assert_allclose(test, [[0.5, 0], [2, 0], [2, 0]])
    np.testing.assert_array_equal(labels, [[0, 0], [1, 1], [1, 1]])
    assert metadata['source_rows'] == {'train': 2, 'test': 3}
    assert metadata['feature_names'] == ['SENSOR_A', 'SENSOR_B']
    assert metadata['source_files']['train'].endswith('SWaT_Dataset_Normal_v1.xlsx')


@pytest.mark.parametrize('dataset,create', [('WADI', wadi_release), ('SWaT', swat_release)])
def test_downsampling_uses_identical_test_label_indices(tmp_path, dataset, create):
    root = tmp_path / dataset
    create(root)
    full, sampled = tmp_path / 'full', tmp_path / 'sampled'
    preprocess_industrial(dataset, root, full)
    metadata = preprocess_industrial(dataset, root, sampled, downsample=2)
    for expected, actual in zip(arrays(full), arrays(sampled)):
        np.testing.assert_array_equal(actual, expected[::2])
    assert metadata['downsample'] == 2


@pytest.mark.parametrize('dataset,create,bad_labels', [
    ('WADI', wadi_release, (1, 0, -1)),
    ('WADI', wadi_release, (1, '', -1)),
    ('SWaT', swat_release, ('Normal', 'Unknown', 'Attack')),
])
def test_unknown_labels_fail_instead_of_becoming_normal(tmp_path, dataset, create, bad_labels):
    create(tmp_path / dataset, bad_labels)
    with pytest.raises(ValueError, match='label'):
        preprocess_industrial(dataset, tmp_path / dataset, tmp_path / 'processed')
    assert not (tmp_path / 'processed' / 'labels.npy').exists()


@pytest.mark.parametrize('stride', [0, -1, True, 1.5])
def test_invalid_downsample_rejected(tmp_path, stride):
    with pytest.raises(ValueError, match='positive integer'):
        preprocess_industrial('WADI', tmp_path, tmp_path, downsample=stride)


def test_sensor_mismatch_rejected(tmp_path):
    import pandas as pd
    with pytest.raises(ValueError, match='sensor columns disagree'):
        _prepare_features(pd.DataFrame({'A': [1, 2]}), pd.DataFrame({'B': [1, 2]}))


def test_preprocessing_dispatch_uses_configured_roots(monkeypatch, tmp_path):
    wadi_release(tmp_path / 'data' / 'WADI')
    monkeypatch.setattr(preprocess, 'data_folder', str(tmp_path / 'data'))
    monkeypatch.setattr(preprocess, 'output_folder', str(tmp_path / 'processed'))
    metadata = preprocess.load_data('WADI')
    assert metadata['shapes']['labels'] == [3, 2]
    assert (tmp_path / 'processed' / 'WADI' / 'train.npy').exists()


def test_processed_arrays_load_into_main_and_bind_sensor_identity(monkeypatch, tmp_path):
    wadi_release(tmp_path / 'WADI')
    destination = tmp_path / 'processed' / 'WADI'
    preprocess_industrial('WADI', tmp_path / 'WADI', destination)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, 'output_folder', str(tmp_path / 'processed'))
    monkeypatch.setattr(main, 'DEVICE', torch.device('cpu'))
    monkeypatch.setattr(main, 'DTYPE', torch.float32)
    monkeypatch.setattr(main, 'args', SimpleNamespace(dataset='WADI', model='AdvSTAD',
                                                    less=False, test=False, retrain=True))
    train, test, labels = main.load_dataset('WADI')
    features = main.validate_dataset_shapes(train, test, labels)
    metadata = main.load_dataset_metadata('WADI', features)
    model, _, _, _, _ = main.load_model('AdvSTAD', features, metadata['feature_names'])
    assert train.dtype == test.dtype == torch.float32
    assert checkpoint_metadata(model)['sensor_identifiers'] == ['SENSOR_A', 'SENSOR_B']
    original_key = experiment_key(model, 'WADI')
    model.sensor_identifiers.reverse()
    assert experiment_key(model, 'WADI') != original_key
    with pytest.raises(ValueError, match='metadata disagrees'):
        main.load_dataset_metadata('WADI', 99)


def test_timestamp_labels_skip_localization_and_are_tracked(monkeypatch, tmp_path):
    wadi_release(tmp_path / 'WADI')
    destination = tmp_path / 'processed' / 'WADI'
    preprocess_industrial('WADI', tmp_path / 'WADI', destination)
    config = {
        'device': 'cpu', 'dtype': 'float32',
        'advstad': {'window_size': 2, 'batch_size': 2, 'dropout': 0.0},
        'experiment_tracking': {'enabled': True, 'log_dir': str(tmp_path / 'runs'),
                                'index_file': str(tmp_path / 'experiments.jsonl')},
    }
    arguments = SimpleNamespace(dataset='WADI', model='AdvSTAD', less=False,
                                test=True, retrain=True, run_name='new-release')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, 'output_folder', str(tmp_path / 'processed'))
    monkeypatch.setattr(main, 'CONFIG', config)
    monkeypatch.setattr(main, 'args', arguments)
    monkeypatch.setattr(main, 'DEVICE', torch.device('cpu'))
    monkeypatch.setattr(main, 'DTYPE', torch.float32)
    pot_labels = []
    def fake_pot(reference, score, target):
        pot_labels.append(target.copy())
        return {'f1': 0.0}, np.zeros_like(target)
    monkeypatch.setattr(main, 'pot_eval', fake_pot)
    def unexpected_localization(*args):
        pytest.fail('Timestamp labels cannot establish sensor localization')
    monkeypatch.setattr(main, 'hit_att', unexpected_localization)
    monkeypatch.setattr(main, 'ndcg', unexpected_localization)
    tracker = ExperimentTracker(config, arguments, main.DEVICE, main.DTYPE)
    try:
        main.run_experiment(tracker)
    finally:
        tracker.complete()
    for target in pot_labels:
        np.testing.assert_array_equal(target, [0, 1, 0])
    details = tracker.record['dataset_details']
    assert details['label_scope'] == 'timestamp'
    assert details['attack_timestamps'] == 1
    assert details['preprocessing']['normalization']['method'] == 'training_minmax'
    assert details['sensor_identifiers'] == ['SENSOR_A', 'SENSOR_B']
