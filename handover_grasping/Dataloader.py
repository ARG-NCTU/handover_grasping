# AUTOGENERATED! DO NOT EDIT! File to edit: Dataloader.ipynb (unless otherwise specified).

__all__ = ['handover_grasping_dataset', 'image_net_mean', 'image_net_std']

# Cell
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset
import numpy as np


image_net_mean = np.array([0.485, 0.456, 0.406])
image_net_std  = np.array([0.229, 0.224, 0.225])

class handover_grasping_dataset(Dataset):
    """Dataloader of handover datasets.
    """
    name = []
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])
        if self.mode == 'train':
            f = open(self.data_dir+"/train.txt", "r")
        else:
            f = open(self.data_dir+"/test.txt", "r")
        for _, line in enumerate(f):
              self.name.append(line.replace("\n", ""))

    def __len__(self):
        return len(self.name)
    def __getitem__(self, idx):
        idx_name = self.name[idx]
        color_img = cv2.imread(self.data_dir+"/color/color_"+idx_name+'.png')
        color_img = color_img[:,:,[2,1,0]]
        color_img = cv2.resize(color_img,(224,224))
        color_origin = cv2.resize(color_img,(640,480))
        depth_img = np.load(self.data_dir+"/depth/depth_"+idx_name+'.npy')
        depth_origin = np.load(self.data_dir+"/depth/depth_"+idx_name+'.npy').astype(float)
        depth_img = cv2.resize(depth_img,(224,224))

        if self.mode == 'train':
            label_img = np.load(self.data_dir+"/label/label_"+idx_name+'.npy')

            f = open(self.data_dir+'/idx/id_'+idx_name+'.txt', "r")

            IDX = int(f.readlines()[0])

            # Unlabeled -> 2; unsuctionable -> 0; suctionable -> 1
            label_tmp = np.round(label_img[IDX]/255.*2.).astype(float)
            label = np.zeros((4,28,28))
            # Set label pixel
            label[IDX] = cv2.resize(label_tmp, (int(28), int(28)))
            label[IDX][label[IDX] != 0.0] = 1.0

            label_tensor = self.transform(label).float()

        # uint8 -> float
        color = (color_img/255.).astype(float)
        # BGR -> RGB and normalize
        color_rgb = np.zeros(color.shape)
        for i in range(3):
            color_rgb[:, :, i] = (color[:, :, 2-i]-image_net_mean[i])/image_net_std[i]

        depth_img = np.round((depth_img/np.max(depth_img))*255).astype('int').reshape(1,depth_img.shape[0],depth_img.shape[1])
        depth = (depth_img/1000.).astype(float) # to meters
        # D435 depth range
        depth = np.clip(depth, 0.0, 1.2)
        # Duplicate channel and normalize
        depth_3c = np.zeros(color.shape)
        for i in range(3):
            depth_3c[:, :, i] = (depth[:, :]-image_net_mean[i])/image_net_std[i]


        color_tensor = self.transform(color_rgb).float()
        depth_tensor = self.transform(depth_3c).float()


        if self.mode == 'train':
            sample = {"color": color_tensor, "depth": depth_tensor, "label": label_tensor, "id": 0, "color_origin": color_origin, "depth_origin": depth_origin}
        else:
            sample = {"color": color_tensor, "depth": depth_tensor, "color_origin": color_origin, "depth_origin": depth_origin}

        return sample