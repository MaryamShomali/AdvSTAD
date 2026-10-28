import logging
from numbers import Real

import numpy as np

from src.spot import SPOT
from src.constants import *
from sklearn.metrics import *


_LOGGER = logging.getLogger(__name__)
_POT_MAX_ATTEMPTS = 10


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    try:
        roc_auc = roc_auc_score(actual, predict)
    except:
        roc_auc = 0
    return f1, precision, recall, TP, TN, FP, FN, roc_auc


# the below function is taken from OmniAnomaly code base directly
def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t


def pot_eval(init_score, score, label, q=1e-5, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t
    Returns:
        dict: pot result dict

    Initialization uses lm[0], as before; level is retained for compatibility.
    Invalid inputs raise ValueError. Initialization is limited to 10 attempts
    (9 retries), then raises RuntimeError chained from the last SPOT error.
    """
    lms = lm[0]

    def invalid_input(reason):
        message = (
            f'POT input validation failed (attempts=0, q={q!r}, level={lms!r}): '
            f'{reason}'
        )
        _LOGGER.error(message)
        raise ValueError(message)

    arrays, summaries = [], []
    for name, values in [('init_score', init_score), ('score', score), ('label', label)]:
        try:
            values = np.asarray(values)
        except (TypeError, ValueError) as error:
            invalid_input(f'{name} cannot be converted to an array: {error}')
        if values.ndim != 1 or values.size == 0:
            invalid_input(f'{name} must be a non-empty 1-D array; shape={values.shape}')
        if values.dtype.kind not in 'biuf':
            invalid_input(f'{name} must contain real numeric values; dtype={values.dtype}')
        finite = np.isfinite(values)
        if not finite.all():
            invalid_input(
                f'{name} contains NaN or Inf; size={values.size}, '
                f'nonfinite_count={values.size - np.count_nonzero(finite)}'
            )
        minimum, maximum = values.min(), values.max()
        summaries.append(
            f'{name}(size={values.size}, dtype={values.dtype}, '
            f'min={minimum!r}, max={maximum!r})'
        )
        if name == 'init_score' and minimum == maximum:
            invalid_input(f'{summaries[-1]} is constant; SPOT cannot fit an empty peak set')
        arrays.append(values)
    init_score, score, label = arrays
    if score.size != label.size:
        invalid_input(f'score and label lengths differ: {score.size} != {label.size}')
    if not isinstance(q, Real) or not np.isfinite(q) or not 0 < q < 1:
        invalid_input(f'q must be finite and strictly between 0 and 1; value={q!r}')
    # SPOT normalizes finite levels modulo 1; preserve that existing behavior.
    if not isinstance(lms, Real) or not np.isfinite(lms):
        invalid_input(f'lm[0] must be a finite real number; value={lms!r}')

    diagnostics = '; '.join(summaries)
    for attempt in range(1, _POT_MAX_ATTEMPTS + 1):
        s = None
        try:
            s = SPOT(q)  # SPOT object
            s.fit(init_score, score)  # data import
            s.initialize(level=lms, min_extrema=False, verbose=False)  # initialization step
            if not np.isfinite(s.init_threshold) or not np.isfinite(s.extreme_quantile):
                raise ValueError('SPOT initialization returned a non-finite threshold or quantile')
        except MemoryError:
            raise
        except Exception as error:
            message = (
                f'POT initialization failed (attempt {attempt}/{_POT_MAX_ATTEMPTS}, '
                f'retries={attempt - 1}, q={q!r}, level={lms!r}, '
                f'init_threshold={getattr(s, "init_threshold", None)!r}, '
                f'peaks={getattr(s, "Nt", None)!r}; {diagnostics}): '
                f'{type(error).__name__}: {error}'
            )
            if attempt == _POT_MAX_ATTEMPTS:
                _LOGGER.error(message, exc_info=True)
                raise RuntimeError(
                    f'POT initialization failed after {attempt} attempts '
                    f'({attempt - 1} retries). {message}'
                ) from error
            _LOGGER.warning(message, exc_info=True)
            lms = lms * 0.999
        else:
            if attempt > 1:
                _LOGGER.info(
                    'POT initialization succeeded after %d attempts (%d retries), level=%r',
                    attempt, attempt - 1, lms,
                )
            break
    ret = s.run(dynamic=False)  # run
    # print(len(ret['alarms']))
    # print(len(ret['thresholds']))
    pot_th = np.mean(ret['thresholds']) * lm[1]
    # pot_th = np.percentile(score, 100 * lm[0])
    # np.percentile(score, 100 * lm[0])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    # DEBUG - np.save(f'{debug}.npy', np.array(pred))
    # DEBUG - print(np.argwhere(np.array(pred)))
    p_t = calc_point2point(pred, label)
    # print('POT result: ', p_t, pot_th, p_latency)
    return {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'ROC/AUC': p_t[7],
        'threshold': pot_th,
        # 'pot-latency': p_latency
    }, np.array(pred)
