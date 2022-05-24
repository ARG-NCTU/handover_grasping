import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
from handover_grasping.model import HANet
from handover_grasping.Dataloader import handover_grasping_dataset

parser = argparse.ArgumentParser(description='Set up')
parser.add_argument('--data_dir', type=str, default = '/home/arg/handover_grasping/HANet_datasets')
parser.add_argument('--epoch', type=int, default = 50)
parser.add_argument('--save_every', type=int, default = 25)
parser.add_argument('--batch_size', type=int, default = 5)
args = parser.parse_args()

if os.path.isdir(args.data_dir + '/weight') == False:
    os.mkdir(args.data_dir + '/weight')

dataset = handover_grasping_dataset(args.data_dir, mode='train')
dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)

net = HANet().cuda()

criterion = nn.BCEWithLogitsLoss().cuda()

optimizer = optim.Adam(net.parameters(), lr = 1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 25, gamma = 0.1)

loss_l = []
for epoch in range(args.epoch):
    loss_sum = 0.0
    ts = time.time()
    for i_batch, sampled_batched in enumerate(dataloader):
        print("\r[{:03.2f} %]".format(i_batch/float(len(dataloader))*100.0), end="\r")
        optimizer.zero_grad()
        color = sampled_batched['color'].cuda()
        depth = sampled_batched['depth'].cuda()
        label = sampled_batched['label'].permute(0,2,3,1).cuda().float()
        predict = net(color, depth)

        loss = criterion(predict, label)

        loss.backward()
        loss_sum += loss.detach().cpu().numpy()
        optimizer.step()
    scheduler.step()
    loss_l.append(loss_sum/len(dataloader))
    if (epoch+1)%args.save_every==0:
        torch.save(net.state_dict(), args.data_dir + '/weight/grapnet_{}_{}.pth' .format(epoch+1, round(loss_l[-1],3)))

    print("Epoch: {}| Loss: {}| Time elasped: {}".format(epoch+1, loss_l[-1], time.time()-ts))