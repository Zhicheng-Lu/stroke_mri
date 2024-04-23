from torch import nn
import torch

class Segmentation(nn.Module):
	def __init__(self, data_reader):
		super(Segmentation, self).__init__()
		self.f_size = data_reader.f_size

		# Shared layers between different MRI sequences
		self.shared_down1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.shared_down2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.shared_down3 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.shared_down4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.shared_down5 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=16*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.shared_up1 = nn.Sequential(
			nn.ConvTranspose2d(in_channels=16*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.shared_up2 = nn.Sequential(
			nn.Conv2d(in_channels=16*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.shared_up3 = nn.Sequential(
			nn.Conv2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.shared_up4 = nn.Sequential(
			nn.Conv2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=2*self.f_size, out_channels=self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)


		# DWI
		self.DWI_down1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.DWI_down2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.DWI_down3 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.DWI_down4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.DWI_down5 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=16*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.DWI_up1 = nn.Sequential(
			nn.ConvTranspose2d(in_channels=16*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.DWI_up2 = nn.Sequential(
			nn.Conv2d(in_channels=16*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.DWI_up3 = nn.Sequential(
			nn.Conv2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.DWI_up4 = nn.Sequential(
			nn.Conv2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=2*self.f_size, out_channels=self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.DWI_fuse = nn.Sequential(
			nn.Conv2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)

		# FLAIR
		self.FLAIR_down1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.FLAIR_down2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.FLAIR_down3 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.FLAIR_down4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.FLAIR_down5 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=16*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.FLAIR_up1 = nn.Sequential(
			nn.ConvTranspose2d(in_channels=16*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.FLAIR_up2 = nn.Sequential(
			nn.Conv2d(in_channels=16*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.FLAIR_up3 = nn.Sequential(
			nn.Conv2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.FLAIR_up4 = nn.Sequential(
			nn.Conv2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=2*self.f_size, out_channels=self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.FLAIR_fuse = nn.Sequential(
			nn.Conv2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)

		# T1
		self.T1_down1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.T1_down2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.T1_down3 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.T1_down4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.T1_down5 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=16*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.T1_up1 = nn.Sequential(
			nn.ConvTranspose2d(in_channels=16*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.T1_up2 = nn.Sequential(
			nn.Conv2d(in_channels=16*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.T1_up3 = nn.Sequential(
			nn.Conv2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.T1_up4 = nn.Sequential(
			nn.Conv2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=2*self.f_size, out_channels=self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.T1_fuse = nn.Sequential(
			nn.Conv2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)

		# T2
		self.T2_down1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.T2_down2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.T2_down3 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.T2_down4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.T2_down5 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=16*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU()
		)
		self.T2_up1 = nn.Sequential(
			nn.ConvTranspose2d(in_channels=16*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.T2_up2 = nn.Sequential(
			nn.Conv2d(in_channels=16*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.T2_up3 = nn.Sequential(
			nn.Conv2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.T2_up4 = nn.Sequential(
			nn.Conv2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=2*self.f_size, out_channels=self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.ReLU()
		)
		self.T2_fuse = nn.Sequential(
			nn.Conv2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
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
		self.conv3d_1 = nn.Sequential(
			nn.Conv3d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3,3), padding=(1,1,1)),
			nn.ReLU()
		)
		self.conv3d_2 = nn.Sequential(
			nn.Conv3d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3,3), padding=(1,1,1)),
			nn.ReLU()
		)
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels=2*self.f_size, out_channels=2, kernel_size=(1,1))
		)

		

	def forward(self, device, batch_sequences):
		# Shared layers
		shareds = []
		for sequence in batch_sequences:
			if sequence is None:
				shareds.append(None)
				continue
			down1 = self.shared_down1(sequence)
			down2 = self.shared_down2(down1)
			down3 = self.shared_down3(down2)
			down4 = self.shared_down4(down3)
			down5 = self.shared_down5(down4)
			up1 = self.shared_up1(down5)
			up1_cat = torch.cat((down4, up1), dim=1)
			up2 = self.shared_up2(up1_cat)
			up2_cat = torch.cat((down3, up2), dim=1)
			up3 = self.shared_up3(up2_cat)
			up3_cat = torch.cat((down2, up3), dim=1)
			up4 = self.shared_up4(up3_cat)
			up4_cat = torch.cat((down1, up4), dim=1)
			shareds.append(up4_cat)

		shared_average = torch.mean(torch.stack([sequence for sequence in shareds if sequence is not None]), dim=0)

		# Specific layers
		specific_layers = [{'down1': self.DWI_down1, 'down2': self.DWI_down2, 'down3': self.DWI_down3, 'down4': self.DWI_down4, 'down5': self.DWI_down5, 'up1': self.DWI_up1, 'up2': self.DWI_up2, 'up3': self.DWI_up3, 'up4': self.DWI_up4, 'fuse': self.DWI_fuse},
						   {'down1': self.FLAIR_down1, 'down2': self.FLAIR_down2, 'down3': self.FLAIR_down3, 'down4': self.FLAIR_down4, 'down5': self.FLAIR_down5, 'up1': self.FLAIR_up1, 'up2': self.FLAIR_up2, 'up3': self.FLAIR_up3, 'up4': self.FLAIR_up4, 'fuse': self.FLAIR_fuse},
						   {'down1': self.T1_down1, 'down2': self.T1_down2, 'down3': self.T1_down3, 'down4': self.T1_down4, 'down5': self.T1_down5, 'up1': self.T1_up1, 'up2': self.T1_up2, 'up3': self.T1_up3, 'up4': self.T1_up4, 'fuse': self.T1_fuse},
						   {'down1': self.T2_down1, 'down2': self.T2_down2, 'down3': self.T2_down3, 'down4': self.T2_down4, 'down5': self.T2_down5, 'up1': self.T2_up1, 'up2': self.T2_up2, 'up3': self.T2_up3, 'up4': self.T2_up4, 'fuse': self.T2_fuse}]
		specifics = []
		for i,sequence in enumerate(batch_sequences):
			if sequence is None:
				specifics.append(None)
				continue
			down1 = specific_layers[i]['down1'](sequence)
			down2 = specific_layers[i]['down2'](down1)
			down3 = specific_layers[i]['down3'](down2)
			down4 = specific_layers[i]['down4'](down3)
			down5 = specific_layers[i]['down5'](down4)
			up1 = specific_layers[i]['up1'](down5)
			up1_cat = torch.cat((down4, up1), dim=1)
			up2 = specific_layers[i]['up2'](up1_cat)
			up2_cat = torch.cat((down3, up2), dim=1)
			up3 = specific_layers[i]['up3'](up2_cat)
			up3_cat = torch.cat((down2, up3), dim=1)
			up4 = specific_layers[i]['up4'](up3_cat)
			up4_cat = torch.cat((down1, up4), dim=1)
			specifics.append(up4_cat)

		fused = [layers['fuse'](torch.cat((shared, specific), dim=1)) if specific is not None else shared_average for shared, specific, layers in zip(shareds, specifics, specific_layers)]
		fused = torch.cat(fused, dim=1)
		
		fused = torch.moveaxis(fused, 0, 1)
		fused = self.conv3d_1(fused)
		fused = self.conv3d_2(fused)
		fused = torch.moveaxis(fused, 1, 0)
		output = self.conv(fused)

		# gnn1 = self.GNN_layer(device, fused, self.aggr_1, self.update_1)
		# gnn2 = self.GNN_layer(device, gnn1, self.aggr_2, self.update_2)
		# output = self.conv(gnn2)

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