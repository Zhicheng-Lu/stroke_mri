from torch import nn
import torch

class Segmentation(nn.Module):
	def __init__(self, data_reader):
		super(Segmentation, self).__init__()
		self.f_size = data_reader.f_size

		self.layer_list = nn.ModuleList()
		sequences = ['ADC', 'DWI', 'FLAIR', 'T1', 'T2', 'T1c', 'CBF', 'CBV', 'Tmax', 'TTP']

		self.layers = {}
		for sequence in sequences:
			self.layers[sequence] = {}
			self.layers[sequence]['down1'] = nn.Sequential(
				nn.Conv2d(in_channels=1, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
				nn.ReLU(),
				nn.Conv2d(in_channels=self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
				nn.ReLU()
			)
			self.layers[sequence]['down2'] = nn.Sequential(
				nn.MaxPool2d(kernel_size=(2,2)),
				nn.Conv2d(in_channels=self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
				nn.ReLU(),
				nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
				nn.ReLU()
			)
			self.layers[sequence]['down3'] = nn.Sequential(
				nn.MaxPool2d(kernel_size=(2,2)),
				nn.Conv2d(in_channels=2*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
				nn.ReLU(),
				nn.Conv2d(in_channels=4*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
				nn.ReLU()
			)
			self.layers[sequence]['down4'] = nn.Sequential(
				nn.MaxPool2d(kernel_size=(2,2)),
				nn.Conv2d(in_channels=4*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
				nn.ReLU(),
				nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
				nn.ReLU()
			)
			self.layers[sequence]['down5'] = nn.Sequential(
				nn.MaxPool2d(kernel_size=(2,2)),
				nn.Conv2d(in_channels=8*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3), padding=(1,1)),
				nn.ReLU(),
				nn.Conv2d(in_channels=16*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3), padding=(1,1)),
				nn.ReLU()
			)
			self.layers[sequence]['up1'] = nn.Sequential(
				nn.ConvTranspose2d(in_channels=16*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
				nn.ReLU()
			)
			self.layers[sequence]['up2'] = nn.Sequential(
				nn.Conv2d(in_channels=16*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
				nn.ReLU(),
				nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
				nn.ReLU(),
				nn.ConvTranspose2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
				nn.ReLU()
			)
			self.layers[sequence]['up3'] = nn.Sequential(
				nn.Conv2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
				nn.ReLU(),
				nn.Conv2d(in_channels=4*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
				nn.ReLU(),
				nn.ConvTranspose2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
				nn.ReLU()
			)
			self.layers[sequence]['up4'] = nn.Sequential(
				nn.Conv2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
				nn.ReLU(),
				nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
				nn.ReLU(),
				nn.ConvTranspose2d(in_channels=2*self.f_size, out_channels=self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
				nn.ReLU()
			)
			# self.aggr_1 = nn.Sequential(
			# 	nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			# 	nn.ReLU()
			# )
			# self.update_1 = nn.Sequential(
			# 	nn.Conv2d(in_channels=16*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			# 	nn.ReLU()
			# )
			# self.aggr_2 = nn.Sequential(
			# 	nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			# 	nn.ReLU()
			# )
			# self.update_2 = nn.Sequential(
			# 	nn.Conv2d(in_channels=4*self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			# 	nn.ReLU()
			# )
			self.layers[sequence]['conv3d_1'] = nn.Sequential(
				nn.Conv3d(in_channels=2*self.f_size, out_channels=self.f_size, kernel_size=(3,3,3), padding=(1,1,1)),
				nn.ReLU()
			)
			self.layers[sequence]['conv3d_2'] = nn.Sequential(
				nn.Conv3d(in_channels=self.f_size, out_channels=self.f_size, kernel_size=(3,3,3), padding=(1,1,1)),
				nn.ReLU()
			)
			self.layers[sequence]['dense'] = nn.Sequential(
				nn.Conv2d(in_channels=self.f_size, out_channels=2, kernel_size=(1,1))
			)

		for sequence_name in self.layers:
			for layer_name in self.layers[sequence_name]:
				self.layer_list.append(self.layers[sequence_name][layer_name])

		

	def forward(self, device, batch_sequences):
		output = {}

		for sequence_name, sequence in batch_sequences.items():
			down1 = self.layers[sequence_name]['down1'](sequence)
			down2 = self.layers[sequence_name]['down2'](down1)
			down3 = self.layers[sequence_name]['down3'](down2)
			down4 = self.layers[sequence_name]['down4'](down3)
			down5 = self.layers[sequence_name]['down5'](down4)
			up1 = self.layers[sequence_name]['up1'](down5)
			up1_cat = torch.cat((down4, up1), dim=1)
			up2 = self.layers[sequence_name]['up2'](up1_cat)
			up2_cat = torch.cat((down3, up2), dim=1)
			up3 = self.layers[sequence_name]['up3'](up2_cat)
			up3_cat = torch.cat((down2, up3), dim=1)
			up4 = self.layers[sequence_name]['up4'](up3_cat)
			up4_cat = torch.cat((down1, up4), dim=1)
			
			feature = torch.moveaxis(up4_cat, 0, 1)
			feature = self.layers[sequence_name]['conv3d_1'](feature)
			feature = self.layers[sequence_name]['conv3d_2'](feature)
			feature = torch.moveaxis(feature, 1, 0)
			feature = self.layers[sequence_name]['dense'](feature)
			
			output[sequence_name] = feature

		return output


	def GNN_layer(self, device, in_features, aggr, update):
		features = in_features[:, None, :, :, :]
		features = features.repeat(1,2,1,1,1)


		for n1 in range(len(in_features)):
			similarities = []
			# Find similarities of all pairs
			for n2 in range(len(in_features)):
				if n1 == n2:
					continue
				similarities.append((n2, nn.functional.cosine_similarity(torch.flatten(features[n1,0]), torch.flatten(features[n2,0]), dim=0)))
			# Sort and slice top 5
			similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
			if len(similarities) < 3:
				knn = [similarity[0] for similarity in similarities]
			else:
				knn = [similarities[i][0] for i in range(len(similarities)) if i<3]
			knn = features[knn,0]
			knn = aggr(knn)
			knn = torch.mean(knn, dim=0, keepdim=False)
			features[n1,1] = knn

		features = torch.reshape(features, (features.shape[0], features.shape[1]*features.shape[2], features.shape[3], features.shape[4]))
		features = update(features)


		return features