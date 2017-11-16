import time

import numpy as np
from pycocotools.cocoeval import COCOeval

############################################################
#  COCO Evaluation
############################################################


def build_coco_results(dataset, image_ids, rois, class_ids, scores):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)

            result = {
                'image_id': image_id,
                'category_id': dataset.get_source_class_id(class_id, 'coco'),
                'bbox': [bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0]],
                'score': score,
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type='bbox', limit=0):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: 'bbox' or 'segm' for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]['id'] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i+1],
                                           r['rois'], r['class_ids'],
                                           r['scores'])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    coco_eval = COCOeval(coco, coco_results, eval_type)
    coco_eval.params.imgIds = coco_image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print('Prediction time: {}. Average {}/image'.format(
        t_prediction, t_prediction/len(image_ids)))
    print('Total time: ', time.time() - t_start)
