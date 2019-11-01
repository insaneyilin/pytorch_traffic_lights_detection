import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def parse_label_file(label_file_path):
    label_list = []
    with open(label_file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            fields = line.split()
            color_label = int(fields[0])
            xmin = int(float(fields[1]))
            ymin = int(float(fields[2]))
            xmax = int(float(fields[3]))
            ymax = int(float(fields[4]))
            label_dict = {'color_label': color_label,\
                    'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
            label_list.append(label_dict)
    return label_list


class TLDataset(torch.utils.data.Dataset):
    def __init__(self, list_file_path):
        dataset_root_dir = os.path.dirname(list_file_path)
        self.dataset_root_dir = dataset_root_dir
        self.all_labels = []
        with open(list_file_path, 'r') as f:
            for line in f.readlines():
                fields = line.strip().split()
                image_path = fields[0]
                label_path = fields[1]
                image_path = os.path.abspath(os.path.join(dataset_root_dir, image_path))
                label_path = os.path.abspath(os.path.join(dataset_root_dir, label_path))
                image_label = {'path': image_path, 'boxes': []}
                image_label['boxes'] = parse_label_file(label_path)
                self.all_labels.append(image_label)
        self.transforms = transforms.Compose([transforms.ToTensor()])


    def __getitem__(self, index):
        img_path = self.all_labels[index]['path']
        img = Image.open(img_path).convert("RGB")
        boxes = []
        for box in self.all_labels[index]['boxes']:
            xmin = box['xmin']
            ymin = box['ymin']
            xmax = box['xmax']
            ymax = box['ymax']
            boxes.append([xmin, ymin, xmax, ymax])
        num_boxes = len(boxes)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_boxes,), dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_boxes,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return self.transforms(img), target


    def __len__(self):
        return len(self.all_labels)


