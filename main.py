import pickle
import os
import sys
import inspect
import json
import shutil
from collections.abc import Mapping
from uuid import uuid4
import pandas as pd
import torch
from tqdm import tqdm
from src.config import CONFIG, DEVICE, DTYPE, get_advstad_config
from src.advstad_training import build_optimizers, train_epoch, evaluate
from src.advstad_checkpoint import experiment_key, checkpoint_payload, restore_checkpoint
from src.experiment_tracking import ExperimentTracker
from src.memory import is_out_of_memory_error, release_memory
from src.runtime_measurement import synchronized_time
from src.study_protocol import (
	ADVSTAD_MODELS, INCLUSIVE_ALIGNMENT, LEGACY_ALIGNMENT, SMD_MACHINE_ID,
	dataset_provenance, window_alignment,
)
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import DataLoader
import torch.nn as nn
from time import perf_counter
from pprint import pprint
# from beepy import beep

def convert_to_windows(data, model):
	if data.ndim != 2 or data.shape[0] == 0:
		raise ValueError('Window conversion requires a non-empty 2-D tensor.')

	w_size = model.n_window
	n_rows, n_feats = data.shape
	padding_size = w_size - 1 if window_alignment(model.name) == INCLUSIVE_ALIGNMENT else w_size
	padding = data[0:1].expand(padding_size, -1)
	padded = torch.cat((padding, data), dim=0)
	row_stride, feature_stride = padded.stride()

	# Each window is a read-only overlapping view into one padded tensor. This
	# avoids allocating an n_window-times-larger copy of the entire dataset.
	if model.name in ADVSTAD_MODELS or 'TranAD' in model.name or 'Attention' in model.name:
		return padded.as_strided(
			(n_rows, w_size, n_feats),
			(row_stride, row_stride, feature_stride),
		)
	return padded.as_strided(
		(n_rows, w_size * n_feats),
		(row_stride, feature_stride),
	)

def load_dataset(dataset):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise FileNotFoundError(f'Processed data not found. Run: python preprocess.py {dataset}')
	loaded = {}
	for split in ['train', 'test', 'labels']:
		file = split
		if dataset == 'SMD': file = SMD_MACHINE_ID + '_' + file
		if dataset == 'SMAP': file = 'P-1_' + file
		if dataset == 'MSL': file = 'C-1_' + file
		if dataset == 'UCR': file = '136_' + file
		if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
		values = np.load(os.path.join(folder, f'{file}.npy'))
		if split == 'train' and args.less:
			values = cut_array(0.2, values)
		if split == 'labels':
			# Evaluation is NumPy-based, so labels remain on the host.
			loaded[split] = values
		else:
			loaded[split] = torch.from_numpy(values).to(
				device=DEVICE,
				dtype=DTYPE,
			)
		del values
	return loaded['train'], loaded['test'], loaded['labels']


def load_dataset_metadata(dataset, feature_count):
	path = os.path.join(output_folder, dataset, 'metadata.json')
	if not os.path.exists(path):
		return {}
	with open(path, encoding='utf-8') as handle:
		metadata = json.load(handle)
	if len(metadata.get('feature_names', [])) != feature_count:
		raise ValueError(f'Processed feature metadata disagrees with the arrays in {path}. Re-run preprocessing.')
	return metadata

def model_experiment_key(model):
	if model.name in ADVSTAD_MODELS:
		return experiment_key(model, args.dataset)
	return f'{args.model}_{args.dataset}'


def validate_dataset_shapes(train_data, test_data, labels):
	"""Resolve sensors from observations before constructing a model."""
	for name, values in [('train', train_data), ('test', test_data), ('labels', labels)]:
		if values.ndim != 2 or min(values.shape) == 0:
			raise ValueError(f'{name} must be a non-empty [timestamps, sensors] array.')
	if train_data.shape[1] != test_data.shape[1] or test_data.shape != labels.shape:
		raise ValueError(
			'Train/test sensor counts and test/label shapes must agree: '
			f'train={tuple(train_data.shape)}, test={tuple(test_data.shape)}, '
			f'labels={tuple(labels.shape)}.'
		)
	return train_data.shape[1]


def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{model_experiment_key(model)}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	temporary_path = f'{file_path}.tmp'
	try:
		# Keep the pre-invocation checkpoint before the first write, including
		# fresh --retrain runs. Later epochs still update the usual resume path.
		if not getattr(model, '_checkpoint_write_prepared', False):
			if os.path.exists(file_path):
				archive_folder = os.path.join(folder, 'archive')
				os.makedirs(archive_folder, exist_ok=True)
				archive_path = os.path.join(archive_folder, f'model-{uuid4().hex}.ckpt')
				with open(file_path, 'rb') as source, open(archive_path, 'xb') as target:
					shutil.copyfileobj(source, target)
				model.previous_checkpoint_archive = archive_path
			model._checkpoint_write_prepared = True
		payload = checkpoint_payload(model, optimizer, scheduler, epoch, accuracy_list) if model.name in ADVSTAD_MODELS else {
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'scheduler_state_dict': scheduler.state_dict(),
			'accuracy_list': accuracy_list}
		if model.name == 'TranAD_Corrected':
			payload['model'] = model.name
			payload['window_alignment'] = INCLUSIVE_ALIGNMENT
		# Optional provenance: old payloads remain loadable without this field.
		payload['training_seed'] = getattr(model, 'training_seed', torch.initial_seed())
		torch.save(payload, temporary_path)
		os.replace(temporary_path, file_path)
	except BaseException:
		try:
			os.remove(temporary_path)
		except OSError:
			pass
		raise
	return file_path


def validate_test_checkpoint(checkpoint, path, model_name):
	if (
		not isinstance(checkpoint, Mapping)
		or isinstance(checkpoint.get('epoch'), bool)
		or not isinstance(checkpoint.get('epoch'), int)
		or checkpoint['epoch'] < 0
		or not isinstance(checkpoint.get('model_state_dict'), Mapping)
		or not checkpoint['model_state_dict']
	):
		raise ValueError(
			f'--test requires a valid trained checkpoint with at least one completed epoch: {path}'
		)
	required = (
		{'metadata', 'schema_version', 'optimizer_state_dicts', 'scheduler_state_dicts', 'accuracy_list', 'training_history'}
		if model_name in ADVSTAD_MODELS
		else {'optimizer_state_dict', 'scheduler_state_dict', 'accuracy_list'}
	)
	missing = required - set(checkpoint)
	if missing:
		raise ValueError(f'Invalid trained checkpoint required by --test: {path}; missing {sorted(missing)}')


def load_model(modelname, dims, sensor_identifiers=None):
	import src.models
	model_class = getattr(src.models, modelname)
	if modelname in ADVSTAD_MODELS:
		resolved_config = get_advstad_config(CONFIG, dims, lr_d[args.dataset])
		model = model_class(dims, resolved_config).to(device=DEVICE, dtype=DTYPE)
		if sensor_identifiers is not None:
			model.sensor_identifiers = list(sensor_identifiers)
		optimizer, scheduler = build_optimizers(model)
	else:
		model = model_class(dims).to(device=DEVICE, dtype=DTYPE)
		optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	model.training_seed = torch.initial_seed()
	fname = f'checkpoints/{model_experiment_key(model)}/model.ckpt'
	if args.test and not os.path.isfile(fname):
		raise FileNotFoundError(
			f'--test requires an existing trained checkpoint: {fname}. '
			'Run training explicitly before requesting test-only evaluation.'
		)
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		load_options = (
			{'weights_only': True}
			if modelname in ADVSTAD_MODELS and 'weights_only' in inspect.signature(torch.load).parameters
			else {}
		)
		try:
			checkpoint = torch.load(fname, map_location=DEVICE, **load_options)
		except (OSError, RuntimeError, ValueError, pickle.UnpicklingError, EOFError) as error:
			if not args.test:
				raise
			raise ValueError(f'Could not load the trained checkpoint required by --test: {fname}: {error}') from error
		if args.test:
			validate_test_checkpoint(checkpoint, fname, modelname)
		if isinstance(checkpoint, Mapping):
			saved_seed = checkpoint.get('training_seed')
			if saved_seed is not None and saved_seed != torch.initial_seed():
				raise ValueError(
					f'Checkpoint training seed {saved_seed} differs from the selected seed '
					f'{torch.initial_seed()}: {fname}. Use the matching seed or --retrain for a fresh training run.'
				)
			model.training_seed = saved_seed
		if modelname in ADVSTAD_MODELS:
			epoch, accuracy_list = restore_checkpoint(checkpoint, model, optimizer, scheduler)
			return model, optimizer, scheduler, epoch, accuracy_list
		if modelname == 'TranAD_Corrected' and (
			checkpoint.get('model') != modelname
			or checkpoint.get('window_alignment') != INCLUSIVE_ALIGNMENT
		):
			raise ValueError('Incompatible TranAD_Corrected checkpoint: model/alignment metadata differs. Use --retrain.')
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		for parameter, state in optimizer.state.items():
			for key, value in state.items():
				if torch.is_tensor(value):
					target_dtype = parameter.dtype if value.is_floating_point() else value.dtype
					state[key] = value.to(device=parameter.device, dtype=target_dtype)
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, feats, optimizer, scheduler, training = True):
	if model.name in ADVSTAD_MODELS:
		if training:
			metrics = train_epoch(model, data, optimizer, scheduler, epoch)
			model.last_training_metrics = metrics
			if not hasattr(model, 'training_history'):
				model.training_history = []
			model.training_history.append({'epoch': epoch, **metrics})
			tqdm.write(f'Epoch {epoch},\tGenerator = {metrics["loss_g"]},\tAdversary = {metrics["loss_a"]}')
			return metrics['loss_g'], metrics['lr_g']
		return evaluate(model, data)
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	if 'DAGMM' in model.name:
		l = nn.MSELoss(reduction = 'none')
		compute = ComputeLoss(model, 0.1, 0.005, DEVICE, model.n_gmm)
		n = epoch + 1; w_size = model.n_window
		l1s = []; l2s = []
		if training:
			for d in data:
				_, x_hat, z, gamma = model(d)
				l1, l2 = l(x_hat, d), l(gamma, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1) + torch.mean(l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s = []
			for d in data: 
				_, x_hat, _, _ = model(d)
				ae1s.append(x_hat)
			ae1s = torch.stack(ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
	if 'Attention' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []; res = []
		if training:
			for d in data:
				ae, ats = model(d)
				# res.append(torch.mean(ats, axis=0).view(-1))
				l1 = l(ae, d)
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			# res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			ae1s, y_pred = [], []
			for d in data: 
				ae1 = model(d)
				y_pred.append(ae1[-1])
				ae1s.append(ae1)
			ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
			loss = torch.mean(l(ae1s, data), axis=1)
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
	elif 'OmniAnomaly' in model.name:
		if training:
			mses, klds = [], []
			for i, d in enumerate(data):
				y_pred, mu, logvar, hidden = model(d, hidden if i else None)
				MSE = l(y_pred, d)
				KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
				loss = MSE + model.beta * KLD
				mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			y_preds = []
			for i, d in enumerate(data):
				y_pred, _, _, hidden = model(d, hidden if i else None)
				y_preds.append(y_pred)
			y_pred = torch.stack(y_preds)
			MSE = l(y_pred, data)
			return MSE.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
	elif 'USAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d in data:
				ae1s, ae2s, ae2ae1s = model(d)
				l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
				l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1 + l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s, ae2s, ae2ae1s = [], [], []
			for d in data: 
				ae1, ae2, ae2ae1 = model(d)
				ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
			ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
	elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []
		if training:
			for i, d in enumerate(data):
				if 'MTAD_GAT' in model.name: 
					x, h = model(d, h if i else None)
				else:
					x = model(d)
				loss = torch.mean(l(x, d))
				l1s.append(torch.mean(loss).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			xs = []
			for d in data: 
				if 'MTAD_GAT' in model.name: 
					x, h = model(d, None)
				else:
					x = model(d)
				xs.append(x)
			xs = torch.stack(xs)
			y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(xs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
	elif 'GAN' in model.name:
		l = nn.MSELoss(reduction = 'none')
		bcel = nn.BCELoss(reduction = 'mean')
		msel = nn.MSELoss(reduction = 'mean')
		real_label = torch.tensor([0.9], dtype=DTYPE, device=DEVICE)
		fake_label = torch.tensor([0.1], dtype=DTYPE, device=DEVICE) # label smoothing
		n = epoch + 1; w_size = model.n_window
		mses, gls, dls = [], [], []
		if training:
			for d in data:
				# training discriminator
				model.discriminator.zero_grad()
				_, real, fake = model(d)
				dl = bcel(real, real_label) + bcel(fake, fake_label)
				dl.backward()
				model.generator.zero_grad()
				optimizer.step()
				# training generator
				z, _, fake = model(d)
				mse = msel(z, d) 
				gl = bcel(fake, real_label)
				tl = gl + mse
				tl.backward()
				model.discriminator.zero_grad()
				optimizer.step()
				mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
				# tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
			return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
		else:
			outputs = []
			for d in data: 
				z, _, _ = model(d)
				outputs.append(z)
			outputs = torch.stack(outputs)
			y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(outputs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
	elif 'TranAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		data_x = data.to(dtype=DTYPE)
		bs = model.batch
		dataloader = DataLoader(data_x, batch_size=bs)
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d in dataloader:
				d = d.to(DEVICE)
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			losses, predictions = [], []
			for d in dataloader:
				d = d.to(DEVICE)
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				if isinstance(z, tuple): z = z[1]
				losses.append(l(z, elem)[0].detach().cpu())
				predictions.append(z[0].detach().cpu())
			return torch.cat(losses).numpy(), torch.cat(predictions).numpy()
	else:
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()

def run_experiment(tracker):
	trainD, testD, labels = load_dataset(args.dataset)
	feats = validate_dataset_shapes(trainD, testD, labels)
	dataset_metadata = load_dataset_metadata(args.dataset, feats)
	tracker.log_dataset(trainD, testD, labels, metadata=dataset_metadata)
	tracker.log_data_provenance(dataset_provenance(args.dataset))
	if dataset_metadata.get('label_scope') == 'timestamp':
		print('Labels identify attack timestamps; sensor-localization metrics are omitted.')
	if args.model in ['MERLIN']:
		result, evaluation_time = eval(f'run_{args.model.lower()}(testD, labels, args.dataset)')
		tracker.log_overall_evaluation(result)
		tracker.log_timing('evaluation', evaluation_time)
		return
	model, optimizer, scheduler, epoch, accuracy_list = load_model(
		args.model, feats, sensor_identifiers=dataset_metadata.get('feature_names'),
	)

	## Prepare data
	windowed = (
		model.name in [*ADVSTAD_MODELS, 'Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN']
		or 'TranAD' in model.name
	)
	num_epochs = model.resolved_config['training']['epochs_per_run'] if model.name in ADVSTAD_MODELS else 5
	artifact_key = model_experiment_key(model)
	# Store new plots under an invocation-specific directory so previous run
	# records continue pointing to their own immutable plot files.
	run_id = tracker.record['run_id'] if tracker.enabled else f'untracked-{uuid4().hex}'
	plot_key = f'{artifact_key}/{run_id}'
	tracker.log_model(model, optimizer, scheduler, epoch + 1, windowed, num_epochs)
	checkpoint_path = f'checkpoints/{artifact_key}/model.ckpt'
	if epoch >= 0:
		tracker.log_checkpoint(checkpoint_path, epoch, loaded=True)
	tracker.log_runtime(training_epochs_total=epoch + 1)
	testO = testD if not args.test else None
	if windowed and not args.test:
		trainD = convert_to_windows(trainD, model)

	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		e = epoch + 1; start = perf_counter()
		compute_seconds = 0.0
		completed_epochs = 0
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			epoch_start = synchronized_time(DEVICE)
			lossT, lr = backprop(e, model, trainD, feats, optimizer, scheduler)
			compute_seconds += synchronized_time(DEVICE) - epoch_start
			completed_epochs += 1
			tracker.log_runtime(
				training_total_seconds=compute_seconds,
				training_average_epoch_seconds=compute_seconds / completed_epochs,
				training_epochs_this_run=completed_epochs,
				training_epochs_total=e + 1,
			)
			accuracy_list.append((lossT, lr))
			checkpoint_path = save_model(model, optimizer, scheduler, e, accuracy_list)
			if hasattr(model, 'previous_checkpoint_archive'):
				tracker.log_artifact('previous_checkpoint_archive', model.previous_checkpoint_archive)
			if model.name in ADVSTAD_MODELS:
				tracker.log_training_epoch(e, lossT, lr, metrics=model.last_training_metrics)
			else:
				tracker.log_training_epoch(e, lossT, lr)
			tracker.log_checkpoint(checkpoint_path, e)
		training_time = perf_counter() - start
		print(color.BOLD+'Training time: '+"{:10.4f}".format(training_time)+' s'+color.ENDC)
		tracker.log_timing('training', training_time)
		if model.name in ADVSTAD_MODELS:
			plot_accuracies(accuracy_list, plot_key, loss_label='Generator objective')
		else:
			plot_accuracies(accuracy_list, plot_key)
		tracker.log_artifact(
			'training_plot', f'plots/{plot_key}/training-graph.pdf',
		)

	### Testing phase
	evaluation_start = perf_counter()
	if windowed:
		testD = convert_to_windows(testD, model)
	model.eval()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	with torch.no_grad():
		inference_start = synchronized_time(DEVICE)
		loss, y_pred = backprop(0, model, testD, feats, optimizer, scheduler, training=False)
		inference_seconds = synchronized_time(DEVICE) - inference_start
	tracker.log_runtime(
		inference_total_seconds=inference_seconds,
		inference_latency_ms_per_timepoint=1000.0 * inference_seconds / len(loss),
	)
	tracker.log_timing('inference', inference_seconds)
	del testD

	### Plot curves
	if not args.test:
		if window_alignment(model.name) == LEGACY_ALIGNMENT: testO = torch.roll(testO, 1, 0)
		plotter(plot_key, testO, y_pred, loss, labels)
		tracker.log_artifact('evaluation_plot', f'plots/{plot_key}/output.pdf')
		del testO
	release_memory()

	### Scores
	if windowed and args.test:
		trainD = convert_to_windows(trainD, model)
	feature_results = []
	with torch.no_grad():
		lossT, _ = backprop(0, model, trainD, feats, optimizer, scheduler, training=False)
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		result, pred = pot_eval(lt, l, ls); preds.append(pred)
		feature_results.append(result)
	df = pd.DataFrame(feature_results)
	# preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
	# pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
	if dataset_metadata.get('label_scope') != 'timestamp':
		result.update(hit_att(loss, labels))
		result.update(ndcg(loss, labels))
	tracker.log_evaluation(lossT, loss, result, df)
	tracker.log_timing('evaluation', perf_counter() - evaluation_start)
	tracker.log_resource_usage(DEVICE)
	print(df)
	pprint(result)
	# pprint(getresults2(df, result))
	# beep(4)


def main():
	tracker = ExperimentTracker(CONFIG, args, DEVICE, DTYPE)
	try:
		run_experiment(tracker)
	except BaseException as error:
		try:
			tracker.log_resource_usage(DEVICE)
		except Exception as resource_error:
			print(f'Could not record resource usage: {resource_error}', file=sys.stderr)
		try:
			tracker.fail(error)
		except Exception as tracking_error:
			print(f'Could not record experiment failure: {tracking_error}', file=sys.stderr)
		raise
	tracker.complete()
	if tracker.enabled:
		print(f'Experiment artifacts: {tracker.run_dir}')


def run_with_memory_guard():
	try:
		main()
		return 0
	except (MemoryError, OSError, RuntimeError) as error:
		if not is_out_of_memory_error(error):
			raise
		details = str(error)

	# At this point the failed operation's traceback and tensors are no longer
	# retained by the exception handler, so cached accelerator memory can be freed.
	release_memory()
	print(
		f'{color.FAIL}Out of memory. The current run was stopped safely and memory '
		f'caches were released.{color.ENDC}',
		file=sys.stderr,
	)
	print(
		'Any checkpoint from the last completed epoch is still available. '
		'Try --less, a smaller model/batch, or device: "cpu" in config.yaml.',
		file=sys.stderr,
	)
	if details:
		print(f'Details: {details}', file=sys.stderr)
	return 1


if __name__ == '__main__':
	raise SystemExit(run_with_memory_guard())
