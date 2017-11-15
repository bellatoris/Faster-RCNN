import os

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils import data
from torchvision import transforms


class CocoDetection(data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform

        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def load_coco(self, subset):
        """Load a subset of the COCO dataset.
        subset: What to load(train, val, minival, val35k)
        """
        # Path
        image_dir = os.path.join(self.root, 'images',
                                 'train2014' if subset == 'train'
                                 else 'val2014')

        #  Create COCO object
        json_path_dict = {
            'train': 'annotations/instances_train2014.json',
            'val': 'annotations/instances_val2014.json',
            'minival': 'annotations/instances_minival2014.json',
            'val35k': 'annotations/instances_valminusminival2014.json',
        }

        coco = COCO(os.path.join(self.root, json_path_dict[subset]))
        class_ids = sorted(coco.getCatIds())
        image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class('coco', i, coco.loadCats(i)[0]['name'])

        # Add images
        for i in image_ids:
            ann_ids = coco.getAnnIds(imgIds=[i], iscrowd=False)
            if not ann_ids:
                continue

            self.add_image('coco', image_id=i,
                           path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                           width=coco.imgs[i]['width'],
                           height=coco.imgs[i]['height'],
                           annotations=coco.loadAnns(ann_ids))

    def add_class(self, source, class_id, class_name):
        assert '.' not in source, 'Source name cannot contain a dot'
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info['id'] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            'source': source,
            'id': class_id,
            'name': class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            'id': image_id,
            'source': source,
            'path': path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def __getitem__(self, index):
        target = self.image_info[index]['annotations']
        path = self.image_info[index]['path']

        img = Image.open(path).convert('RGB')
        img, scale = resize(img, 600)
        img = transforms.to_tensor(img)
        img = self.transform(img)

        for t in target:
            x1, y1, w, h = t['bbox']
            x2 = x1 + w
            y2 = y1 + h
            t['bbox'] = np.array([y1, x1, y2, x2]) * scale

        return img, target

    def __len__(self):
        return len(self.image_info)


def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. The smaller edge of the
            image will be matched to this number maintaining the aspect ratio.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img, 1
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation), ow / w
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation), ow / w
    else:
        raise TypeError
