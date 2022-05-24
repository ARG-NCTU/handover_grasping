from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import cv2
from handover_grasping.model import HANet
from handover_grasping.Dataloader import handover_grasping_dataset
from handover_grasping.utils import get_grasp_line, get_affordancemap

dataset = handover_grasping_dataset('/home/arg/handover_grasping/HANet_sample_datasets', mode='test')
dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 8)


net = HANet(pretrained=True).cuda()
net.eval()


for i, batch in enumerate(dataloader):
    color = batch['color'].cuda()
    depth = batch['depth'].cuda()

    predict = net(color, depth)
    predict = predict.cpu().detach().numpy()

    Depth_origin = batch['depth_origin'].cpu().detach().numpy()[0]
    Color_origin = batch['color_origin'].cpu().detach().numpy()[0]

    affordancemap, x, y, theta = get_affordancemap(predict, Depth_origin)

    Combine = cv2.addWeighted(Color_origin,0.7,affordancemap, 0.3,0)

    point1, point2 = get_grasp_line(theta, [y, x], Depth_origin)

    Combine = cv2.line(Combine,point1,point2,(0,0,255),3)
    Combine = cv2.circle(Combine, (int(x), int(y)), 5, (0,255,0), -1)
    Combine = cv2.circle(Combine, point1, 5, (0,255,255), -1)
    Combine = cv2.circle(Combine, point2, 5, (255,255,0), -1)

    plt.imshow(Combine)
    plt.show()