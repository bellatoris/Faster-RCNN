def get_coco():
    import os

    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    root_dir = os.path.expanduser('~/dataset/coco')
    data_type = 'val2017'
    data_dir = '{}/images/{}'.format(root_dir, data_type)
    ann_file = '{}/annotations/instances_{}.json'.format(root_dir,
                                                         data_type)

    det = dset.CocoDetection(root=data_dir,
                             annFile=ann_file,
                             transform=transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.RandomCrop(224),
                                 transforms.ToTensor(),
                             ]))

    print('Number of samples: {}'.format(len(det)))
    img, target = det[3]  # load 4th sample

    print('Image size: {}'.format(img.size()))
    print(target)

    data_loader = DataLoader(det,
                             batch_size=128,
                             shuffle=True,
                             num_workers=4)

    return data_loader


if __name__ == '__main__':
    get_coco()


def coco_api_prac():
    import os

    from pycocotools.coco import COCO
    import numpy as np
    import skimage.io as io
    import matplotlib.pyplot as plt
    import pylab

    pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    data_dir= os.path.expanduser('~/dataset/coco')
    data_type = 'val2017'
    ann_file = '{}/annotations/instances_{}.json'.format(data_dir,
                                                         data_type)
    coco = COCO(ann_file)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercatgories: \n{}'.format(' '.join(nms)))

    # get all images containing given categories, select one at random
    cat_ids = coco.getCatIds(catNms=['person', 'dog', 'skateboard'])
    img_ids = coco.getImgIds(catIds=cat_ids)
    img_ids = coco.getImgIds(imgIds=[324158])
    img = coco.loadImgs(img_ids[np.random.randint(0, len(img_ids))])[0]

    # load and display image
    # I = io.imread('%s/images/%s/%s'%(data_dir, data_type, img['file_name']))
    I = io.imread(img['coco_url'])
    plt.axis('off')
    plt.imshow(I)
    plt.show()

    # load and display instance annotations
    plt.imshow(I)
    plt.axis('off')
    ann_ids = coco.getAnnIds(imgIds=img['id'],
                             catIds=cat_ids,
                             iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    coco.showAnns(anns)
    plt.show()
