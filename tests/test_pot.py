"""POT failure boundaries and exact compatibility with successful legacy scoring."""

import logging

import numpy as np
import pytest

from src import pot


def legacy_pot_eval(init_score, score, label, q=1e-5):
    """Original successful path, with a test-only guard against fixture hangs."""
    lms = pot.lm[0]
    for _ in range(100):
        try:
            s = pot.SPOT(q)
            s.fit(init_score, score)
            s.initialize(level=lms, min_extrema=False, verbose=False)
        except Exception:
            lms = lms * 0.999
        else:
            break
    else:
        pytest.fail('Legacy reference did not initialize within 100 attempts')
    ret = s.run(dynamic=False)
    threshold = np.mean(ret['thresholds']) * pot.lm[1]
    prediction, _ = pot.adjust_predicts(score, label, threshold, calc_latency=True)
    metrics = pot.calc_point2point(prediction, label)
    names = ('f1', 'precision', 'recall', 'TP', 'TN', 'FP', 'FN', 'ROC/AUC')
    return {**dict(zip(names, metrics)), 'threshold': threshold}, np.array(prediction)


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.setattr(pot, 'lm', (0.993, 1))
    rng = np.random.RandomState(42)
    calibration = rng.exponential(size=6000)
    score = rng.exponential(size=300)
    label = np.zeros(300, dtype=np.uint8)
    score[40:50] += 10
    label[40:50] = 1
    return calibration, score, label


def assert_identical(actual, expected):
    assert actual[0] == expected[0]  # Every metric and threshold, without tolerance.
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('lm', [(0.993, 1), (0.99995, 1.06), (0.87, 1), (0, 1), (1.25, 1)])
def test_success_is_identical_to_legacy_and_does_not_mutate_inputs(inputs, monkeypatch, dtype, lm):
    monkeypatch.setattr(pot, 'lm', lm)
    calibration, score, label = inputs
    arrays = (calibration.astype(dtype), score.astype(dtype), label)
    before = [array.copy() for array in arrays]
    expected = legacy_pot_eval(*arrays, q=1e-4)
    # The historically unused level argument must not replace lm[0].
    actual = pot.pot_eval(*arrays, q=1e-4, level=0.4)
    assert_identical(actual, expected)
    for array, original in zip(arrays, before):
        np.testing.assert_array_equal(array, original)
        assert array.dtype == original.dtype


def test_constant_test_scores_and_labels_remain_valid(inputs):
    calibration, score, label = inputs
    score = np.zeros_like(score)
    label = np.zeros_like(label)
    assert_identical(pot.pot_eval(calibration, score, label), legacy_pot_eval(calibration, score, label))


def test_lists_keep_legacy_results(inputs):
    arrays = [inputs[0].tolist(), inputs[1].tolist(), inputs[2]]
    assert_identical(pot.pot_eval(*arrays), legacy_pot_eval(*arrays))


def test_tied_tail_recovers_at_exact_legacy_levels(inputs, monkeypatch, caplog):
    _, score, label = inputs
    calibration = np.concatenate((np.linspace(0.01, 1, 9900), np.full(100, 2.0)))
    levels = []
    original = pot.SPOT.initialize

    def record_level(self, level, **kwargs):
        levels.append(level)
        return original(self, level=level, **kwargs)

    monkeypatch.setattr(pot.SPOT, 'initialize', record_level)
    expected = legacy_pot_eval(calibration, score, label)
    legacy_levels = levels[:]
    levels.clear()
    with caplog.at_level(logging.INFO, logger=pot.__name__):
        actual = pot.pot_eval(calibration, score, label)
    assert_identical(actual, expected)
    assert levels == legacy_levels
    assert 1 < len(levels) <= pot._POT_MAX_ATTEMPTS
    assert len([record for record in caplog.records if record.levelno == logging.WARNING]) == len(levels) - 1
    assert 'initialization succeeded after' in caplog.text


def test_transient_initialization_errors_preserve_legacy_results(inputs, monkeypatch, caplog):
    calls = []
    original = pot.SPOT.initialize

    def transient(self, level, **kwargs):
        calls.append(level)
        if len(calls) <= 2:
            raise RuntimeError('injected calibration error')
        return original(self, level=level, **kwargs)

    monkeypatch.setattr(pot.SPOT, 'initialize', transient)
    expected = legacy_pot_eval(*inputs)
    legacy_levels = calls[:]
    calls.clear()
    with caplog.at_level(logging.INFO, logger=pot.__name__):
        actual = pot.pot_eval(*inputs)
    assert_identical(actual, expected)
    assert calls == legacy_levels
    assert 'RuntimeError: injected calibration error' in caplog.text
    assert 'attempt 1/10, retries=0' in caplog.text
    assert 'succeeded after 3 attempts (2 retries)' in caplog.text


@pytest.mark.parametrize('value', [0.0, 1.0, -2.5])
def test_constant_calibration_is_rejected_without_initialization(inputs, monkeypatch, caplog, value):
    _, score, label = inputs

    def unexpected(*args, **kwargs):
        pytest.fail('Constant calibration must not reach SPOT')

    monkeypatch.setattr(pot, 'SPOT', unexpected)
    with pytest.raises(ValueError, match='init_score.*constant.*empty peak set'):
        pot.pot_eval(np.full(100, value), score, label)
    assert 'attempts=0' in caplog.text
    assert 'size=100' in caplog.text
    assert f'min={value!r}, max={value!r}' in caplog.text


@pytest.mark.parametrize('index,name', [(0, 'init_score'), (1, 'score'), (2, 'label')])
@pytest.mark.parametrize('invalid,reason', [
    ([], 'non-empty 1-D'),
    (1.0, 'non-empty 1-D'),
    ([[1.0, 2.0]], 'non-empty 1-D'),
    ([0.0, np.nan], 'NaN or Inf'),
    ([0.0, np.inf], 'NaN or Inf'),
    ([0.0, -np.inf], 'NaN or Inf'),
    (['a', 'b'], 'real numeric'),
    ([0.0, 1j], 'real numeric'),
    ([None, 1], 'real numeric'),
])
def test_invalid_arrays_fail_before_spot(inputs, monkeypatch, caplog, index, name, invalid, reason):
    arrays = list(inputs)
    arrays[index] = invalid

    def unexpected(*args, **kwargs):
        pytest.fail('Invalid input must not reach SPOT')

    monkeypatch.setattr(pot, 'SPOT', unexpected)
    with pytest.raises(ValueError, match=f'{name}.*{reason}'):
        pot.pot_eval(*arrays)
    assert 'POT input validation failed (attempts=0' in caplog.text
    assert 'q=1e-05, level=0.993' in caplog.text


def test_mismatched_labels_fail_before_spot(inputs, caplog):
    calibration, score, label = inputs
    with pytest.raises(ValueError, match='lengths differ: 300 != 299'):
        pot.pot_eval(calibration, score, label[:-1])
    assert 'attempts=0' in caplog.text


@pytest.mark.parametrize('invalid', [0, 1, -0.1, np.nan, np.inf, None, '0.993'])
def test_invalid_risk_fails_before_initialization(inputs, invalid):
    with pytest.raises(ValueError, match='must be finite and strictly between 0 and 1'):
        pot.pot_eval(*inputs, q=invalid)


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf, None, '0.993'])
def test_invalid_initial_level_fails_before_initialization(inputs, monkeypatch, invalid):
    monkeypatch.setattr(pot, 'lm', (invalid, 1))
    with pytest.raises(ValueError, match=r'lm\[0\] must be a finite real number'):
        pot.pot_eval(*inputs)


def test_real_empty_peak_errors_are_bounded_logged_and_chained(inputs, monkeypatch, caplog):
    _, score, label = inputs
    # Nonconstant calibration with a large tied maximum: all ten attempts have
    # empty peaks. Unlike constant calibration, this exercises actual SPOT errors.
    calibration = np.concatenate((np.zeros(10), np.ones(90)))
    levels = []
    original = pot.SPOT.initialize

    def record_level(self, level, **kwargs):
        levels.append(level)
        return original(self, level=level, **kwargs)

    def unexpected_run(*args, **kwargs):
        pytest.fail('Failed initialization must not produce fallback predictions')

    monkeypatch.setattr(pot.SPOT, 'initialize', record_level)
    monkeypatch.setattr(pot.SPOT, 'run', unexpected_run)
    with pytest.raises(RuntimeError, match=r'after 10 attempts \(9 retries\)') as caught:
        pot.pot_eval(calibration, score, label)
    assert isinstance(caught.value.__cause__, ValueError)
    assert 'zero-size array' in str(caught.value.__cause__)
    assert 'zero-size array' in str(caught.value)
    assert 'init_score(size=100, dtype=float64, min=0.0, max=1.0)' in str(caught.value)
    expected_level = 0.993
    assert len(levels) == pot._POT_MAX_ATTEMPTS == 10
    for level in levels:
        assert level == expected_level
        expected_level = expected_level * 0.999
    assert len(caplog.records) == 10
    assert all(record.exc_info is not None for record in caplog.records)
    assert all('peaks=0' in record.message for record in caplog.records)
    assert all('q=1e-05' in record.message for record in caplog.records)
    assert caplog.records[-1].levelno == logging.ERROR
    assert 'attempt 10/10, retries=9' in caplog.text


@pytest.mark.parametrize('stage', ['constructor', 'fit'])
def test_setup_errors_have_bounded_retries_and_original_cause(inputs, monkeypatch, stage):
    calls = []
    error = RuntimeError(f'injected {stage} error')

    def fail(*args, **kwargs):
        calls.append(1)
        raise error

    if stage == 'constructor':
        monkeypatch.setattr(pot, 'SPOT', fail)
    else:
        monkeypatch.setattr(pot.SPOT, 'fit', fail)
    with pytest.raises(RuntimeError, match=f'RuntimeError: injected {stage} error') as caught:
        pot.pot_eval(*inputs)
    assert len(calls) == 10
    assert caught.value.__cause__ is error


@pytest.mark.parametrize('attribute', ['init_threshold', 'extreme_quantile'])
@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_nonfinite_initialization_results_raise_instead_of_scoring(inputs, monkeypatch, attribute, value):
    calls = []

    def nonfinite(self, **kwargs):
        calls.append(1)
        self.init_threshold = 0.5
        self.extreme_quantile = 1.0
        setattr(self, attribute, value)

    monkeypatch.setattr(pot.SPOT, 'initialize', nonfinite)
    with pytest.raises(RuntimeError, match='non-finite threshold or quantile') as caught:
        pot.pot_eval(*inputs)
    assert isinstance(caught.value.__cause__, ValueError)
    assert len(calls) == 10


@pytest.mark.parametrize('error', [KeyboardInterrupt(), SystemExit(2), MemoryError('out of memory')])
def test_interrupts_and_memory_errors_are_not_retried(inputs, monkeypatch, error):
    calls = []

    def fail(self, **kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr(pot.SPOT, 'initialize', fail)
    with pytest.raises(type(error)) as caught:
        pot.pot_eval(*inputs)
    assert caught.value is error
    assert len(calls) == 1
