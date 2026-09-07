import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _timestamp():
	return datetime.now().astimezone().isoformat(timespec='seconds')


def _path_from_project(path):
	path = Path(path).resolve()
	try:
		return str(path.relative_to(PROJECT_ROOT))
	except ValueError:
		return str(path)


def _json_safe(value):
	if isinstance(value, dict):
		return {str(key): _json_safe(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_json_safe(item) for item in value]
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, np.generic):
		return _json_safe(value.item())
	if isinstance(value, float) and not math.isfinite(value):
		return None
	if isinstance(value, (str, int, float, bool)) or value is None:
		return value
	return str(value)


def _metric_tag(name):
	name = name.lower().replace('roc/auc', 'roc_auc')
	name = name.replace('@', '_at_').replace('%', '_percent')
	return re.sub(r'[^a-z0-9_]+', '_', name).strip('_')


def _git_metadata():
	try:
		revision = subprocess.run(
			['git', 'rev-parse', 'HEAD'],
			cwd=str(PROJECT_ROOT),
			check=True,
			capture_output=True,
			text=True,
		).stdout.strip()
		dirty = bool(subprocess.run(
			['git', 'status', '--porcelain'],
			cwd=str(PROJECT_ROOT),
			check=True,
			capture_output=True,
			text=True,
		).stdout.strip())
		return {'revision': revision, 'dirty': dirty}
	except (OSError, subprocess.SubprocessError):
		return None


class ExperimentTracker:
	"""Write TensorBoard events and a portable JSON record for one run."""

	def __init__(self, config, args, device, dtype):
		tracking_config = config.get('experiment_tracking', {})
		if tracking_config is None:
			tracking_config = {}
		if not isinstance(tracking_config, dict):
			raise ValueError("config.yaml 'experiment_tracking' must be a mapping")

		self.enabled = tracking_config.get('enabled', True)
		if not isinstance(self.enabled, bool):
			raise ValueError("config.yaml 'experiment_tracking.enabled' must be true or false")
		self.writer = None
		self.closed = False
		self.record = None
		self.run_dir = None
		if not self.enabled:
			return

		log_root = Path(tracking_config.get('log_dir', 'runs'))
		if not log_root.is_absolute():
			log_root = PROJECT_ROOT / log_root
		index_file = Path(tracking_config.get('index_file', 'experiments.jsonl'))
		if not index_file.is_absolute():
			index_file = PROJECT_ROOT / index_file

		started_at = _timestamp()
		compact_time = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%f%z')
		run_suffix = '-'.join(filter(None, (
			getattr(args, 'run_name', None), args.model, args.dataset,
		)))
		run_suffix = re.sub(r'[^A-Za-z0-9_.-]+', '-', run_suffix).strip('-')
		run_id = f'{compact_time}-{run_suffix}'
		self.run_dir = log_root / run_id
		tensorboard_dir = self.run_dir / 'tensorboard'
		self.summary_path = self.run_dir / 'run.json'
		self.feature_metrics_path = self.run_dir / 'feature_metrics.csv'
		self.index_file = index_file

		try:
			from torch.utils.tensorboard import SummaryWriter
		except ImportError as error:
			raise RuntimeError(
				'TensorBoard tracking is enabled but tensorboard is not installed. '
				'Install requirements.txt or disable experiment_tracking in config.yaml.'
			) from error
		tensorboard_dir.mkdir(parents=True, exist_ok=False)
		self.writer = SummaryWriter(log_dir=str(tensorboard_dir))

		dtype_name = str(dtype).replace('torch.', '')
		label = getattr(args, 'run_name', None) or device.type.upper()
		self.record = {
			'schema_version': 1,
			'run_id': run_id,
			'date': started_at[:10],
			'label': label,
			'model': args.model,
			'dataset': args.dataset,
			'status': 'running',
			'timestamps': {
				'started_at': started_at,
				'completed_at': None,
			},
			'mode': {
				'test_only': args.test,
				'retrain': args.retrain,
				'less_data': args.less,
			},
			'command': sys.argv,
			'configuration': config,
			'environment': {
				'python': platform.python_version(),
				'platform': platform.platform(),
				'hostname': platform.node(),
				'pytorch': torch.__version__,
				'requested_device': config.get('device'),
				'device': str(device),
				'dtype': dtype_name,
				'cuda_available': torch.cuda.is_available(),
			},
			'git': _git_metadata(),
			'dataset_details': {},
			'hyperparameters': {},
			'checkpoint': {},
			'metrics': {
				'training': [],
				'evaluation': {},
			},
			'artifacts': {
				'run_directory': _path_from_project(self.run_dir),
				'tensorboard_directory': _path_from_project(tensorboard_dir),
				'summary': _path_from_project(self.summary_path),
				'feature_metrics': _path_from_project(self.feature_metrics_path),
				'experiment_index': _path_from_project(self.index_file),
			},
		}
		if device.type == 'cuda':
			self.record['environment']['cuda_device'] = torch.cuda.get_device_name(device)
		self.writer.add_text('experiment/configuration', json.dumps(
			_json_safe(config), indent=2, sort_keys=True,
		))
		self._write_summary()

	def _write_summary(self):
		if not self.enabled:
			return
		temporary_path = self.summary_path.with_suffix('.json.tmp')
		with open(temporary_path, 'w', encoding='utf-8') as summary_file:
			json.dump(
				_json_safe(self.record), summary_file, indent=2, sort_keys=True,
				allow_nan=False,
			)
			summary_file.write('\n')
		os.replace(temporary_path, self.summary_path)

	def log_dataset(self, train_data, test_data, labels):
		if not self.enabled:
			return
		anomaly_count = int(np.count_nonzero(labels))
		details = {
			'train_shape': list(train_data.shape),
			'test_shape': list(test_data.shape),
			'labels_shape': list(labels.shape),
			'time_steps': int(labels.shape[0]),
			'features': int(labels.shape[1]),
			'anomalous_labels': anomaly_count,
			'anomalous_label_rate': anomaly_count / labels.size,
		}
		self.record['dataset_details'] = details
		if self.record['model'] == 'AdvSTAD':
			# Sigmoid outputs are retained, but observations are never clipped.
			details['observation_ranges'] = {
				name: {'min': float(values.min()), 'max': float(values.max())}
				for name, values in [('train', train_data), ('test', test_data)]
			}
		self.writer.add_scalar('dataset/train_time_steps', train_data.shape[0], 0)
		self.writer.add_scalar('dataset/test_time_steps', test_data.shape[0], 0)
		self.writer.add_scalar('dataset/features', labels.shape[1], 0)
		self.writer.add_scalar('dataset/anomalous_label_rate', details['anomalous_label_rate'], 0)
		self._write_summary()

	def log_model(self, model, optimizer, scheduler, starting_epoch, windowed, epochs_per_run):
		if not self.enabled:
			return
		model_attributes = {}
		for name in (
			'lr', 'batch', 'n_window', 'n_feats', 'n_hidden', 'n_latent',
			'n_gmm', 'beta',
		):
			if hasattr(model, name):
				model_attributes[name] = getattr(model, name)
		def optimizer_details(value):
			return {
				'name': value.__class__.__name__,
				'learning_rate': value.param_groups[0]['lr'],
				'weight_decay': value.param_groups[0].get('weight_decay'),
			}

		def scheduler_details(value):
			return {
				'name': value.__class__.__name__,
				'step_size': getattr(value, 'step_size', None),
				'gamma': getattr(value, 'gamma', None),
			}

		named_optimizers = isinstance(optimizer, dict)
		hyperparameters = {
			'epochs_per_run': epochs_per_run,
			'model': model_attributes,
			'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
			'trainable_parameter_count': sum(
				parameter.numel() for parameter in model.parameters() if parameter.requires_grad
			),
			'starting_epoch': starting_epoch,
			'windowed_input': windowed,
		}
		if named_optimizers:
			hyperparameters['optimizers'] = {name: optimizer_details(value) for name, value in optimizer.items()}
			hyperparameters['schedulers'] = {name: scheduler_details(value) for name, value in scheduler.items()}
		else:
			hyperparameters['optimizer'] = optimizer_details(optimizer)
			hyperparameters['scheduler'] = scheduler_details(scheduler)
		if model.name == 'AdvSTAD':
			from src.advstad_checkpoint import checkpoint_metadata, experiment_key
			metadata = checkpoint_metadata(model)
			hyperparameters['advstad'] = model.resolved_config
			hyperparameters['checkpoint_metadata'] = metadata
			hyperparameters['training_loss_label'] = 'Generator objective'
			self.record['experiment_key'] = experiment_key(model, self.record['dataset'])
			self.record['dataset_details']['window_alignment'] = metadata['window_alignment']
			self.record['dataset_details']['sensor_identifiers'] = metadata['sensor_identifiers']
		elif 'TranAD' in model.name:
			self.record['dataset_details']['window_alignment'] = 'legacy_previous_endpoint'
		self.record['hyperparameters'] = hyperparameters
		self.writer.add_text('experiment/hyperparameters', json.dumps(
			_json_safe(hyperparameters), indent=2, sort_keys=True,
		))
		self._write_summary()

	def log_checkpoint(self, checkpoint_path, epoch, loaded=False):
		if not self.enabled:
			return
		self.record['checkpoint'] = {
			'path': _path_from_project(checkpoint_path),
			'epoch': epoch,
			'loaded': loaded,
		}
		self._write_summary()

	def log_artifact(self, name, path):
		if not self.enabled:
			return
		self.record['artifacts'][name] = _path_from_project(path)
		self._write_summary()

	def log_training_epoch(self, epoch, loss, learning_rate, metrics=None):
		if not self.enabled:
			return
		extra_metrics = metrics or {}
		metrics = {
			'epoch': epoch,
			'loss': loss,
			'learning_rate': learning_rate,
			**extra_metrics,
		}
		self.record['metrics']['training'].append(metrics)
		self.writer.add_scalar('training/loss', loss, epoch)
		self.writer.add_scalar('training/learning_rate', learning_rate, epoch)
		for name, value in extra_metrics.items():
			self.writer.add_scalar(f'training/{_metric_tag(name)}', value, epoch)
		self._write_summary()

	def log_timing(self, name, seconds):
		if not self.enabled:
			return
		self.record['metrics'].setdefault('timing_seconds', {})[name] = seconds
		self.writer.add_scalar(f'timing/{_metric_tag(name)}_seconds', seconds, 0)
		self._write_summary()

	def log_resource_usage(self, device):
		"""Record process peaks with explicit scope for cost comparisons."""
		if not self.enabled:
			return
		usage = {}
		try:
			import resource
			rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
			usage['process_peak_rss_bytes'] = int(rss if sys.platform == 'darwin' else rss * 1024)
		except ImportError:
			pass
		if device.type == 'cuda':
			usage['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
			usage['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
		self.record['metrics']['memory'] = usage
		for name, value in usage.items():
			self.writer.add_scalar(f'memory/{name}', value, 0)
		self._write_summary()

	def log_evaluation(self, reference_loss, test_loss, result, feature_metrics):
		if not self.enabled:
			return
		feature_records = []
		for feature, row in feature_metrics.iterrows():
			record = {'feature': int(feature)}
			record.update(row.to_dict())
			feature_records.append(record)
		feature_metrics.to_csv(self.feature_metrics_path, index_label='feature')

		evaluation = {
			'reference_anomaly_score': {
				'mean': np.mean(reference_loss),
				'std': np.std(reference_loss),
				'min': np.min(reference_loss),
				'max': np.max(reference_loss),
			},
			'test_anomaly_score': {
				'mean': np.mean(test_loss),
				'std': np.std(test_loss),
				'min': np.min(test_loss),
				'max': np.max(test_loss),
			},
			'overall': result,
			'per_feature': feature_records,
		}
		self.record['metrics']['evaluation'] = evaluation
		for split in ('reference_anomaly_score', 'test_anomaly_score'):
			for name, value in evaluation[split].items():
				self.writer.add_scalar(f'evaluation/{split}/{name}', value, 0)
		for name, value in result.items():
			if isinstance(value, (int, float, np.number)) and np.isfinite(value):
				self.writer.add_scalar(f'evaluation/overall/{_metric_tag(name)}', value, 0)
		for name in feature_metrics.columns:
			for feature, value in feature_metrics[name].items():
				if isinstance(value, (int, float, np.number)) and np.isfinite(value):
					self.writer.add_scalar(
						f'evaluation/per_feature/{_metric_tag(name)}', value, int(feature),
					)
		self._write_summary()

	def log_overall_evaluation(self, result):
		if not self.enabled:
			return
		self.record['metrics']['evaluation']['overall'] = result
		for name, value in result.items():
			if isinstance(value, (int, float, np.number)) and np.isfinite(value):
				self.writer.add_scalar(f'evaluation/overall/{_metric_tag(name)}', value, 0)
		self._write_summary()

	def complete(self):
		if not self.enabled or self.closed:
			return
		self.record['status'] = 'completed'
		self.record['timestamps']['completed_at'] = _timestamp()
		self._finalize()

	def fail(self, error):
		if not self.enabled or self.closed:
			return
		self.record['status'] = 'failed'
		self.record['timestamps']['completed_at'] = _timestamp()
		self.record['error'] = {
			'type': error.__class__.__name__,
			'message': str(error),
		}
		self._finalize()

	def _finalize(self):
		self._write_summary()
		self.index_file.parent.mkdir(parents=True, exist_ok=True)
		index_entry = json.dumps(
			_json_safe(self.record), sort_keys=True, allow_nan=False,
		) + '\n'
		with open(self.index_file, 'a', encoding='utf-8') as index_file:
			index_file.write(index_entry)
		self.writer.flush()
		self.writer.close()
		self.closed = True
