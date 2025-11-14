import torch
import matplotlib.pyplot as plt
from .utils import get_class_name

def print_plot(image, predictions, title, confidence_threshold=0.20):
    if isinstance(image, torch.Tensor):
        image_np = image.detach().permute(1,2,0).cpu().numpy()
    else:
        image_np = image
    
    boxes = predictions['boxes'].detach().cpu().numpy()
    labels = predictions['labels'].detach().cpu().numpy()
    scores = predictions['scores'].detach().cpu().numpy()

    plt.figure(figsize=(12,8))
    plt.imshow(image_np)
    plt.title(title)
    plt.axis('off')

    for i in range(len(boxes)):
        if scores[i] > confidence_threshold:
            x1,y1,x2,y2 = boxes[i]
            rectangle = plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color='red',linewidth=2)
            plt.gca().add_patch(rectangle)
            class_name = get_class_name(labels[i])
            label = f'{class_name}'
            plt.text(x1,y1 - 10,label,color='red',fontsize=10,bbox=dict(boxstyle="round,pad=0.3"))
    plt.tight_layout()
    plt.show()

def compare_detections(original_image, adversarial_image, detector, threshold=0.20):
    original_predictions = detector.detect_objects(original_image)
    adversarial_predictions = detector.detect_objects(adversarial_image)    

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,10))

    original_np = original_image.detach().permute(1,2,0).cpu().numpy()
    adversarial_np = adversarial_image.detach().permute(1,2,0).cpu().numpy()

    ax1.imshow(original_np)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(adversarial_np)
    ax2.set_title('Adversarial Image')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    print_plot(original_image, original_predictions, "Original Image Detections", confidence_threshold=threshold)
    print_plot(adversarial_image, adversarial_predictions, "Adversarial Image Detections", confidence_threshold=threshold)