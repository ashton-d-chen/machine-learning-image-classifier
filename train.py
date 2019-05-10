import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torchvision.models as models
import time
import argparse

def build_data(data_dir):
    train_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.RandomHorizontalFlip(),transforms.RandomRotation(30),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])])
    
    train_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=102, shuffle=True)
    return train_dataloader, train_dataset
    
    
def build_model(model_name, hidden_units, output_size, class_to_idx):
    model = models.densenet121(pretrained=True)
    classifier_input_size = model.classifier.in_features
    if (model_name == 'densenet121'):
        model = models.densenet121(pretrained=True)
        classifier_input_size = model.classifier.in_features
    elif (model_name == 'vgg16'):
        model = models.vgg16(pretrained=True)
        classifier_input_size = model.classifier[0].in_features
        
    for param in model.parameters():
        param.requires_grad = False
    
    classifier_input_size = model.classifier.in_features

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    model.in_features = classifier_input_size
    model.class_to_idx = class_to_idx
    return model


def train(model, data_loader, learning_rate, epochs, gpu):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    criterion = nn.NLLLoss()

    model.optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device);

    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in data_loader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            model.optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            model.optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. ")
                running_loss = 0
                validate(data_loader, model, gpu)

            
def validate(data_loader, model, gpu):
    criterion = nn.NLLLoss()
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model.forward(inputs)
            
            test_loss += criterion(outputs, labels).item()

            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(data_loader):.3f}.. "
          f"Test accuracy: {accuracy/len(data_loader):.3f}")
    model.train()

    
def save(file_path, architecture, input_size, hidden_layers, output_size, optimizer, class_to_idx, state_dict):
    model.cpu()
    checkpoint = {'architecture': architecture,
                  'input_size': input_size,
                  'hidden_layers': hidden_layers,
                  'output_size': output_size,
                  'class_to_idx': class_to_idx,
                  'optimizer' : optimizer.state_dict(),
                  'state_dict': state_dict}
    torch.save(checkpoint, file_path)
    

    
if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('data_directory')
    parser.add_argument('--save_dir', default='./checkpoint.pth')
    parser.add_argument('--arch', default='densenet121')
    parser.add_argument('--learning_rate', default=0.003)
    parser.add_argument('--hidden_units', default=512)
    parser.add_argument('--epochs', default=1)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    data_directory = args.data_directory
    save_dir = args.save_dir
    architecture = args.arch
    learning_rate =args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu
    output_size = len(next(os.walk(data_directory))[1])


    data_loader, data_set = build_data(data_directory)
    model = build_model(architecture, hidden_units, output_size, data_set.class_to_idx)
    train(model, data_loader, learning_rate, epochs, gpu)

    save(save_dir, 
         architecture, 
         model.in_features, 
         hidden_units, 
         output_size,
         model.optimizer,
         model.class_to_idx,
         model.state_dict())

    end_time = time.time()
#     print(end_time - start_time)


