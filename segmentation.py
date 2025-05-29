import os
import glob
import cv2
import numpy as np
import torch
from torch import nn
from data_reader import DataReader
from torch.cuda import amp
from models.segmentation import Segmentation
import nibabel as nib



def segmentation_train(data_reader, device, time):
	# Define loss and model
	entropy_loss_fn = nn.CrossEntropyLoss()
	dice_loss_fn = Diceloss()
	model = Segmentation(data_reader)
	model = model.to(device)

	# Define optimier and scaler
	optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)
	scaler = torch.cuda.amp.GradScaler(enabled=amp)

	losses = []

	os.mkdir(f'checkpoints/segmentation_model_{time}')

	for epoch in range(data_reader.segmentation_epochs):
		optimizer.zero_grad(set_to_none=True)
		torch.cuda.empty_cache()
		# Train
		train_set = [patient for patient in data_reader.segmentation_folders['train']]
		test_set = [patient for patient in data_reader.segmentation_folders['test']]

		for iteration, sequence_dict in enumerate(train_set):
			torch.cuda.empty_cache()
			batch_sequences, masks = data_reader.read_in_batch_segmentation(sequence_dict)

			batch_sequences = {sequence_name: torch.from_numpy(np.moveaxis(sequence, 3, 1)).to(device=device, dtype=torch.float) for sequence_name, sequence in batch_sequences.items()}
			masks = torch.from_numpy(masks)
			masks = masks.type(torch.cuda.LongTensor)
			masks.to(device)

			model.train()
			with torch.cuda.amp.autocast():
				pred = model(device, batch_sequences)
				loss = calculate_loss(pred, masks, entropy_loss_fn, dice_loss_fn)

			print(f"Epoch {epoch+1} iteration {iteration+1} loss: {loss.item()}")
			f = open(f'checkpoints/segmentation_model_{time}/training.txt', 'a')
			f.write(f"Epoch {epoch+1} iteration {iteration+1} loss: {loss.item()}\n")
			f.close()

			# Backpropagation
			scaler.scale(loss).backward()

			# Gradient accumulation
			if (iteration+1) % data_reader.batch_size == 0:
				scaler.step(optimizer)
				optimizer.zero_grad(set_to_none=True)
				scaler.update()

		torch.save(model.state_dict(), f'checkpoints/segmentation_model_{time}/epoch_{str(epoch+1).zfill(3)}.pt')

		# Test on train set
		train_loss = 0.0
		dataset_losses = {'ATLAS2': {'total': 0.0, 'count': 0}, 'ISLES2022': {'total': 0.0, 'count': 0}, 'SISS': {'total': 0.0, 'count': 0}, 'SPES': {'total': 0.0, 'count': 0}}
		for iteration, sequence_dict in enumerate(train_set):
			batch_sequences, masks = data_reader.read_in_batch_segmentation(sequence_dict)

			batch_sequences = {sequence_name: torch.from_numpy(np.moveaxis(sequence, 3, 1)).to(device=device, dtype=torch.float) for sequence_name, sequence in batch_sequences.items()}
			masks = torch.from_numpy(masks)
			masks = masks.type(torch.cuda.LongTensor)
			masks.to(device)

			with torch.no_grad():
				pred = model(device, batch_sequences)
				loss = calculate_loss(pred, masks, entropy_loss_fn, dice_loss_fn)
				train_loss += loss.item()

				dataset = (list(sequence_dict.values())[0]).split('/')[4]
				dataset_losses[dataset]['total'] += loss.item()
				dataset_losses[dataset]['count'] += 1

		train_loss = train_loss / len(train_set)

		losses.append([epoch, train_loss])
		for dataset in ['ATLAS2', 'ISLES2022', 'SISS', 'SPES']:
			if dataset_losses[dataset]['count'] == 0:
				average = 1.0
			else:
				average = dataset_losses[dataset]['total'] / dataset_losses[dataset]['count']
			losses[-1].append(average)

		# Test on test set
		test_loss = 0.0
		dataset_losses = {'ATLAS2': {'total': 0.0, 'count': 0}, 'ISLES2022': {'total': 0.0, 'count': 0}, 'SISS': {'total': 0.0, 'count': 0}, 'SPES': {'total': 0.0, 'count': 0}}
		for iteration, sequence_dict in enumerate(test_set):
			batch_sequences, masks = data_reader.read_in_batch_segmentation(sequence_dict)

			batch_sequences = {sequence_name: torch.from_numpy(np.moveaxis(sequence, 3, 1)).to(device=device, dtype=torch.float) for sequence_name, sequence in batch_sequences.items()}
			masks = torch.from_numpy(masks)
			masks = masks.type(torch.cuda.LongTensor)
			masks.to(device)

			with torch.no_grad():
				pred = model(device, batch_sequences)
				loss = calculate_loss(pred, masks, entropy_loss_fn, dice_loss_fn)
				test_loss += loss.item()

				dataset = (list(sequence_dict.values())[0]).split('/')[4]
				dataset_losses[dataset]['total'] += loss.item()
				dataset_losses[dataset]['count'] += 1

		test_loss = test_loss / len(test_set)

		losses[-1].append(test_loss)
		for dataset in ['ATLAS2', 'ISLES2022', 'SISS', 'SPES']:
			if dataset_losses[dataset]['count'] == 0:
				average = 1.0
			else:
				average = dataset_losses[dataset]['total'] / dataset_losses[dataset]['count']
			losses[-1].append(average)

		[print(f'\t Train {epoch_loss[1]}, test {epoch_loss[6]}') for epoch_loss in losses]

		for epoch, train_loss, ATLAS_train, ISLES2022_train, SISS_train, SPES_train, test_loss, ATLAS_test, ISLES2022_test, SISS_test, SPES_test in losses:
			f = open(f'checkpoints/segmentation_model_{time}/training.txt', 'a')
			f.write(f'Epoch {epoch+1}: training loss {train_loss}, test loss {test_loss}, ATLAS {ATLAS_train} {ATLAS_test}, ISLES2022 {ISLES2022_train} {ISLES2022_test}, SISS {SISS_train} {SISS_test}, SPES {SPES_train} {SPES_test}\n')
			f.close()




def calculate_loss(pred_dict, masks, entropy_loss_fn, dice_loss_fn):
	idv_loss = 0.0
	for sequence in pred_dict.values():
		idv_loss += entropy_loss_fn(sequence, masks) + dice_loss_fn(sequence, masks)

	pred = torch.stack([sequence for sequence in pred_dict.values()])
	pred = torch.mean(pred, dim=0)
	tot_loss = entropy_loss_fn(pred, masks) + dice_loss_fn(pred, masks)

	output = (idv_loss / len(pred_dict) + tot_loss) / 2

	return output





def segmentation_test(data_reader, device, time, visual):
	# Define loss function and model
	entropy_loss_fn = nn.CrossEntropyLoss()
	dice_loss_fn = Diceloss()
	model = Segmentation(data_reader)
	model.load_state_dict(torch.load("checkpoints/segmentation_model.pt"))
	model = model.to(device)

	# Define metrics and initalize empty
	metrics = ['Dice', 'IOU', 'precision', 'recall']
	results = {matrix: {'overall': {'TP': 0, 'FP': 0, 'FN': 0}} for matrix in metrics}

	os.mkdir(f'test/segmentation_{time}')
	f = open(f'test/segmentation_{time}/log.txt', 'a')

	test_set = data_reader.segmentation_folders['test']
	for iteration, sequence_dict in enumerate(test_set):
		batch_sequences, masks = data_reader.read_in_batch_segmentation(sequence_dict)

		dataset = (list(sequence_dict.values())[0]).split('/')[4]
		patient = (list(sequence_dict.values())[0]).split('/')[6]
			
		batch_sequences = {sequence_name: torch.from_numpy(np.moveaxis(sequence, 3, 1)).to(device=device, dtype=torch.float) for sequence_name, sequence in batch_sequences.items()}

		# Infarct level evaluation metrics
		with torch.no_grad():
			pred = model(device, batch_sequences)
			pred = torch.stack([sequence for sequence in pred.values()])
			pred = torch.mean(pred, dim=0)
			pred_softmax = nn.functional.softmax(pred, dim=1).float()
			pred_masks = torch.argmax(pred_softmax, dim=1)
			pred_masks = pred_masks.detach().cpu().numpy()

			# Calculate matrices
			if masks.shape != torch.Size([0]):
				overlap = pred_masks * masks
				area_pred = np.sum(pred_masks)
				area_masks = np.sum(masks)
				TP = np.sum(overlap)
				FP = area_pred - TP
				FN = area_masks - TP
			else:
				TP, FP, FN = 0.0, 0.0, 0.0


		# Write output to file
		os.makedirs(f'test/segmentation_{time}/{dataset}/{patient}')
		f.write(f'{dataset}\t{patient}\n')
		f.write(f'Dice: {(2*TP + 1) / (2*TP + FP + FN + 1)}\n\n')
		for i, pred in enumerate(pred_masks):
			if masks.shape != torch.Size([0]):
				cv2.imwrite(f'test/segmentation_{time}/{dataset}/{patient}/{i}_gt.jpg', masks[i]*255)
			cv2.imwrite(f'test/segmentation_{time}/{dataset}/{patient}/{i}_predicted.jpg', pred*255)

		# original = glob.glob(f'data/datasets/raw/SISS2015_Testing/{patient}/*/*_Flair.*')[0]
		# output_name = original.split('.')[-2]
		# header = nib.load(original).header
		# converted_array = np.array(pred_masks, dtype=np.ushort)
		# affine = np.eye(4)
		# nifti_file = nib.Nifti1Image(converted_array, affine, header)
		# print(output_name)
		# print(header)
		# print(converted_array)
		# nib.save(nifti_file, f'test/segmentation_{time}/{dataset}/VSD.my_result_01.{output_name}.nii')

		# Add dataset to certain metrics
		if not dataset in results['Dice']:
			for matrix in results:
				results[matrix][dataset] = {'TP': 0, 'FP': 0, 'FN': 0}

		
		for matrix in results:
			results[matrix][dataset]['TP'] += TP
			results[matrix][dataset]['FP'] += FP
			results[matrix][dataset]['FN'] += FN
			results[matrix]['overall']['TP'] += TP
			results[matrix]['overall']['FP'] += FP
			results[matrix]['overall']['FN'] += FN

	# Final results
	for matrix in results:
		for dataset in results[matrix]:
			TP = results[matrix][dataset]['TP']
			FP = results[matrix][dataset]['FP']
			FN = results[matrix][dataset]['FN']
			if matrix == 'Dice':
				results[matrix][dataset] = (2*TP + 1) / (2*TP + FP + FN + 1)
			elif matrix == 'IOU':
				results[matrix][dataset] = (TP + 1) / (TP + FP + FN + 1)
			elif matrix == 'precision':
				results[matrix][dataset] = (TP + 1) / (TP + FP + 1)
			elif matrix == 'recall':
				results[matrix][dataset] = (TP + 1) / (TP + FN + 1)
		f.write(f'{matrix}\n{str(results[matrix])}\n')

	f.close()


# Dice Loss Function
class Diceloss(torch.nn.Module):
	def __init__(self):
		super(Diceloss, self).__init__()

	def forward(self, pred, masks):
		pred_softmax = nn.functional.softmax(pred, dim=1).float()
		pred_masks = pred_softmax[:,1,:,:]
		overlap = pred_masks * masks
		area_pred = torch.sum(pred_masks)
		area_masks = torch.sum(masks)
		area_overlap = torch.sum(overlap)

		loss = 1 - (2 * area_overlap + 1) / (area_pred + area_masks + 1)
		return loss