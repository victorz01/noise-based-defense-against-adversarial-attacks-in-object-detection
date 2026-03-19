import torch
import math
from .utils import get_class_name
import math
from .utils import get_class_name
from .noise_functions import add_noise
import numpy as np

def calculate_iou(box1, box2, iou_type="standard"):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    
    iou = intersection / union
    
    if iou_type == "standard":
        return iou
    elif iou_type == "giou":
        return calculate_giou(box1, box2, iou, union)
    elif iou_type == "diou":
        return calculate_diou(box1, box2, iou)
    elif iou_type == "ciou":
        return calculate_ciou(box1, box2, iou)
    else:
        raise ValueError(f"Unknown IoU type: {iou_type}")
    

def calculate_giou(box1, box2, iou, union):
    x1_enc = min(box1[0], box2[0])
    y1_enc = min(box1[1], box2[1])
    x2_enc = max(box1[2], box2[2])
    y2_enc = max(box1[3], box2[3])
    
    enclosing_area = (x2_enc - x1_enc) * (y2_enc - y1_enc)
    
    if enclosing_area == 0:
        return iou
    giou = iou - (enclosing_area - union) / enclosing_area
    return giou

def calculate_diou(box1, box2, iou):
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3])/ 2
    cx2 = (box2[0] + box2[2]) /2
    cy2 = (box2[1] + box2[3]) /2
    center_distance = (cx1 -cx2)**2 + (cy1 -cy2)**2

    x1_enc = min(box1[0],box2[0])
    y1_enc = min(box1[1], box2[1])
    x2_enc=max(box1[2],box2[2])
    y2_enc=max(box1[3],box2[3])
    
    diagonal_distance = (x2_enc - x1_enc) ** 2 + (y2_enc - y1_enc) ** 2
    
    if diagonal_distance == 0:
        return iou
    
    diou = iou - center_distance / diagonal_distance
    return diou

def calculate_ciou(box1, box2, iou):

    diou =calculate_diou(box1, box2, iou)
    w1,h1 = box1[2] -box1[0],box1[3] -box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    
    if h1 == 0 or h2 == 0 or w1 == 0 or w2 == 0:
        return diou
    
    v = (4 / (math.pi ** 2)) * ((math.atan(w1/h1) - math.atan(w2/h2)) ** 2)
    alpha = v / (1 - iou + v) if (1 - iou + v) != 0 else 0
    if iou <= 0:
        alpha = 0
    else:
        alpha = v / (1 - iou + v) if (1 - iou + v) != 0 else 0
    ciou = diou - alpha * v
    return ciou

def compare_detections_iou(predictions1, predictions2, 
                          confidence_threshold,
                          iou_type="standard"):
    '''
    Compares two sets of object detection predictions using one type of IoU.
    '''
    boxes1 = predictions1['boxes'][predictions1['scores'] > confidence_threshold]
    labels1 = predictions1['labels'][predictions1['scores'] > confidence_threshold]
    scores1 = predictions1['scores'][predictions1['scores'] > confidence_threshold]
    boxes2 = predictions2['boxes'][predictions2['scores'] > confidence_threshold]
    labels2 = predictions2['labels'][predictions2['scores'] > confidence_threshold]
    scores2 = predictions2['scores'][predictions2['scores'] > confidence_threshold]

    boxes1 = boxes1.detach().cpu().numpy()
    boxes2 = boxes2.detach().cpu().numpy()
    labels1 = labels1.detach().cpu().numpy()
    labels2 = labels2.detach().cpu().numpy()
    scores1 = scores1.detach().cpu().numpy()
    scores2 = scores2.detach().cpu().numpy()

    sort_idx1 = np.argsort(-scores1) # descending
    boxes1, labels1, scores1 = boxes1[sort_idx1], labels1[sort_idx1], scores1[sort_idx1]
    
    matches = []
    matched_indices_2 = set()
    all_iou_values = [] 
    matched_iou_values = [] 
    
    for i, (box1, label1, score1) in enumerate(zip(boxes1, labels1, scores1)):
        best_iou = -float('inf')
        best_match = None
        
        for j, (box2, label2, score2) in enumerate(zip(boxes2, labels2, scores2)):
            if j in matched_indices_2:
                continue
            if label1 == label2:
                iou = calculate_iou(box1, box2, iou_type)
                all_iou_values.append(iou)  
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = j
        
        if best_match is not None and best_iou > -1.0:
            matches.append({
                'box1_idx': i,
                'box2_idx': best_match,
                'iou': best_iou,
                'label': get_class_name(label1),
                'score1': score1,
                'score2': scores2[best_match]
            })
            matched_iou_values.append(best_iou) 
            matched_indices_2.add(best_match)
    
    num_detections_1 = len(boxes1)
    num_detections_2 = len(boxes2)
    num_matches = len(matches)
    
    precision = num_matches / num_detections_2 if num_detections_2 > 0 else 0

    avg_iou = sum([match['iou'] for match in matches]) / (max(num_detections_1, num_detections_2)) if matches else 0

    return {
        'matches': matches,
        'num_detections_1': num_detections_1,
        'num_detections_2': num_detections_2,
        'num_matches': num_matches,
        'precision': precision,
        'average_iou': avg_iou,
        'iou_type': iou_type,
        'matched_iou_values': matched_iou_values, 
        'all_iou_values': all_iou_values  
    }

def test_noise_defense_with_iou(detector, image_path, target_class, 
                               noise_configs, num_iterations=3,iou_type="standard",confidence_threshold=0.2):
    '''
    Tests the effectiveness of noise as a defense against adversarial patches
    Iterative tests different types of noise 

    '''
    original_image, adversarial_image, _ = detector.generate_adversarial_patch(
        image_path=image_path,
        target_class=target_class,
        num_iterations=num_iterations
    )

    baseline_clean = detector.detect_objects(original_image)
    baseline_adversarial = detector.detect_objects(adversarial_image)
    
    results = {
        'noise_tests': []
    }

    for noise_config in noise_configs:
        noise_type = noise_config['type']
        noise_params = {k: v for k, v in noise_config.items() if k != 'type'}
        noisy_original = add_noise(noise_type, original_image.clone(), **noise_params)
        noisy_adversarial = add_noise(noise_type, adversarial_image.clone(), **noise_params)

        noisy_clean_pred = detector.detect_objects(noisy_original)
        noisy_adversarial_pred = detector.detect_objects(noisy_adversarial)

        clean_comparison = compare_detections_iou(
            baseline_clean, noisy_clean_pred,
            confidence_threshold=confidence_threshold, iou_type=iou_type
        )
        
        adversarial_comparison = compare_detections_iou(
            baseline_adversarial, noisy_adversarial_pred,
            confidence_threshold=confidence_threshold, iou_type=iou_type
        )
        
        results['noise_tests'].append({
            'noise_config': noise_config,
            'clean_comparison': clean_comparison,
            'adversarial_comparison': adversarial_comparison,
        })
    
    return results