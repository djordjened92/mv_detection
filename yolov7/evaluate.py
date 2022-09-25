
import yaml
import argparse
import torch
import torchvision.transforms as T

import test
from multiview_detector.datasets import *

def main(opt):
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)

    base_set = Wildtrack(data_dict['data_root'])
    h0, w0 = base_set.img_shape
    h, w = opt.img_size, opt.img_size
    test_trans = T.Compose([T.Resize([h, w]), T.ToTensor()])
    test_set = frameDataset(base_set, train=False, transform=test_trans, grid_reduce=4)
    testloader = torch.utils.data.DataLoader(test_set,
                                            batch_size=opt.batch_size,
                                            shuffle=False,
                                            num_workers=opt.workers,
                                            pin_memory=True,
                                            collate_fn=frameDataset.collate_fn)

    results, maps, times = test.test(data_dict,
                                     device=opt.device,
                                     weights=opt.weights,
                                     batch_size=opt.batch_size,
                                     imgsz=opt.img_size,
                                     shapes = ((h0, w0), ((h / h0, w / w0), (0., 0.))),
                                     dataloader=testloader,
                                     num_cam=base_set.num_cam)

    # print(f'precision, recall, mAP_0.5, mAP_0.5:0.95, mv_rec, mv_prec, moda, modp, box_loss, obj_loss, cls_loss, mv_loss')
    # print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--weights', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    main(opt)

'''
python evaluate.py \
--data data/data.yaml \
--weights /home/djordje/Documents/Projects/mv_detection/yolov7/runs/train/yolov7-tiny-mv_008/weights/last.pt \
--img-size 832
'''