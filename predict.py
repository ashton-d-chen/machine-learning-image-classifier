import numpy as np
import torch
from torchvision import transforms
import time
from PIL import Image
from matplotlib import pyplot as plt
from train import build_model 
import argparse
import json

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = build_model(checkpoint['architecture'], checkpoint['hidden_layers'], checkpoint['output_size'], checkpoint['class_to_idx'])
    model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     print("Loaded '{}' (arch={}, hidden_layers={})".format(
#     checkpoint_path, 
#     checkpoint['architecture'], 
#     checkpoint['hidden_layers']))
    return model


def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    processed_image = transform(image)
    return processed_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)
        
    image = image.numpy().transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    ax.axis('off')
    ax.imshow(image)

    return ax


def predict(image_path, model, topk, use_gpu):
    from torch.autograd import Variable
    model.eval()
    
    image = Image.open(image_path)
    np_array = process_image(image)
    tensor = np_array
    
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()
        var_inputs = Variable(tensor.float().cuda(), volatile=True)
    else:       
        model = model.cpu()
        var_inputs = Variable(tensor, volatile=True)
    
    var_inputs = var_inputs.unsqueeze(0)
    output = model.forward(var_inputs)  
    ps = torch.exp(output).data.topk(topk)
    probabilities = ps[0].cpu() if use_gpu and torch.cuda.is_available() else ps[0]
    classes = ps[1].cpu() if use_gpu and torch.cuda.is_available() else ps[1]
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    for label in classes.numpy()[0]:
        mapped_classes.append(class_to_idx_inverted[label])
    return probabilities.numpy()[0], mapped_classes


def display_result(probabilities, classes):
    for i in range(len(classes)):
        print(f"{class_to_name[classes[i]]}: {(probabilities[i] * 100):.2f}%")

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('input_path')
    parser.add_argument('checkpoint')
    parser.add_argument('--top_k', default=5)
    parser.add_argument('--category_names', default='./cat_to_name.json')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    input_path = args.input_path
    checkpoint = args.checkpoint
    category_names = args.category_names
    topk = args.top_k
    use_gpu = args.gpu
    
    class_to_name = load_json(category_names)
    
    model = load(checkpoint)

    probabilities, classes = predict(input_path, model, topk, use_gpu)
    
    display_result(probabilities, classes)
        
    end_time = time.time()
#     print(end_time - start_time)