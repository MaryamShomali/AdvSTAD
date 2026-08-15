import pickle
import os
import sys
import pandas as pd
import torch
from tqdm import tqdm
from src.config import DEVICE, DTYPE
from src.memory import is_out_of_memory_error, release_memory
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import DataLoader
import torch.nn as nn
from time import time
from pprint import pprint
# from beepy import beep

def convert_to_windows(data, model):
	if data.ndim != 2 or data.shape[0] == 0:
		raise ValueError('Window conversion requires a non-empty 2-D tensor.')

	w_size = model.n_window
	n_rows, n_feats = data.shape
	padding = data[0:1].expand(w_size, -1)
	padded = torch.cat((padding, data), dim=0)
	row_stride, feature_stride = padded.stride()

	# Each window is a read-only overlapping view into one padded tensor. This
	# avoids allocating an n_window-times-larger copy of the entire dataset.
	if 'TranAD' in model.name or 'Attention' in model.name:
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
		raise Exception('Processed Data not found.')
	loaded = {}
	for split in ['train', 'test', 'labels']:
		file = split
		if dataset == 'SMD': file = 'machine-1-1_' + file
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

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	temporary_path = f'{file_path}.tmp'
	try:
		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'scheduler_state_dict': scheduler.state_dict(),
			'accuracy_list': accuracy_list}, temporary_path)
		os.replace(temporary_path, file_path)
	except BaseException:
		try:
			os.remove(temporary_path)
		except OSError:
			pass
		raise

def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims).to(device=DEVICE, dtype=DTYPE)
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname, map_location=DEVICE)
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

def main():
	trainD, testD, labels = load_dataset(args.dataset)
	if args.model in ['MERLIN']:
		eval(f'run_{args.model.lower()}(testD, labels, args.dataset)')
		return
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

	## Prepare data
	feats = labels.shape[1]
	windowed = (
		model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN']
		or 'TranAD' in model.name
	)
	testO = testD if not args.test else None
	if windowed and not args.test:
		trainD = convert_to_windows(trainD, model)

	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		num_epochs = 5; e = epoch + 1; start = time()
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, feats, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
			save_model(model, optimizer, scheduler, e, accuracy_list)
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

	### Testing phase
	if windowed:
		testD = convert_to_windows(testD, model)
	model.eval()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	with torch.no_grad():
		loss, y_pred = backprop(0, model, testD, feats, optimizer, scheduler, training=False)
	del testD

	### Plot curves
	if not args.test:
		if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0) 
		plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)
		del testO
	release_memory()

	### Scores
	if windowed and args.test:
		trainD = convert_to_windows(trainD, model)
	df = pd.DataFrame()
	with torch.no_grad():
		lossT, _ = backprop(0, model, trainD, feats, optimizer, scheduler, training=False)
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		result, pred = pot_eval(lt, l, ls); preds.append(pred)
		df = df.append(result, ignore_index=True)
	# preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
	# pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
	result.update(hit_att(loss, labels))
	result.update(ndcg(loss, labels))
	print(df)
	pprint(result)
	# pprint(getresults2(df, result))
	# beep(4)


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
