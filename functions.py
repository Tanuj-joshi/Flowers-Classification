import torchvision
import torch
from collections import OrderedDict
from torchvision import transforms
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from matplotlib import pyplot as plt
import random
from PIL import Image
import os, time
import numpy as np
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):

        self.image_dirs = image_dirs
        self.transform = transform
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('jpg')]
            return images
        self.class_names = [i for i in range(1, 103)] 
        self.images = {}
        for class_name in self.class_names:
            self.images[class_name] = get_images(class_name)
    def __len__(self):
        return sum([len(self.images[class_name]) for class_name in self.class_names])
    def __getitem__(self, idx):
        class_name = random.choice(self.class_names)
        idx = idx % len(self.images[class_name])
        image_name = self.images[class_name][idx]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = int(class_name) - 1
        return image, label


def create_dataset(image_dir, train):
    # Image Transformation
    if train is True:
        train_transform = transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])
        valid_transform =  transforms.Compose([transforms.Resize(225),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])
        train_dirs = {}
        for id in range(1,103):
            train_dirs[id] = image_dir +'train/'+str(id)

        val_dirs = {}
        for id in range(1,103):
            val_dirs[id] = image_dir +'valid/'+str(id)
        # Load the datasets with ImageFolder
        # train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        # valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transform)
        train_dataset = ChestXRayDataset(train_dirs, train_transform)
        valid_dataset = ChestXRayDataset(val_dirs, valid_transform)
        return train_dataset, valid_dataset
    else:
        test_transform = transforms.Compose([transforms.Resize(225),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])
        test_dirs = {}
        for id in range(1,103):
            test_dirs[id] = image_dir +str(id)
        test_dataset = ChestXRayDataset(test_dirs, test_transform)
        return test_dataset


def load_model(arch, device, trained_model):
    # load the saved model
    if (arch == "vgg13"):
        input_size = 25088
        hidden_units = 512
        output_size = 102
        model = torchvision.models.vgg13(pretrained=True)
    elif (arch == "densenet121"):
        input_size = 1024
        hidden_units = 512
        output_size = 102
        model = torchvision.models.densenet121(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    #model.class_to_idx = check_path['class_to_idx']
                    
    # Build a feed-forward network at the output end of model
    classifier = torch.nn.Sequential(OrderedDict([('fc1', torch.nn.Linear(input_size, hidden_units)),
                                            ('relu', torch.nn.ReLU()),
                                            ('dropout1',torch.nn.Dropout(0.2)),
                                            ('fc2', torch.nn.Linear(hidden_units, output_size)),
                                            ('output', torch.nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    model = model.to(device)
    if os.path.exists(trained_model):
        if device=="cuda":
            model.load_state_dict(torch.load(trained_model))
        else:
            model.load_state_dict(torch.load(trained_model, map_location=torch.device('cpu')))
    else:
        print("No pretrained model is present in the specified directory..\
              Loading the fresh open-source model")
    return model


# Predict trained model results in new Test Images and display results
def predict(model, optimizer, dataloader, loss_fn, device, train):
    start_time = time.time()
    all_preds = []
    all_labels = []
    total_loss = 0.
    if train is False:
        model.eval()
    else:
        model.train()
    for step, (images,labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        if optimizer!=None:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train==True):  #torch.no_grad():
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() #* images.size(0)
            _, preds = torch.max(outputs, 1)
            if train==True:
                loss.backward()
                optimizer.step()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # for visualizing accuracy and loss for each batch processing
            temp_acc = accuracy_score(all_labels, all_preds)
            temp_loss = total_loss/(step + 1)
            if train is True:
                print(f'Train loss: {temp_loss:.4f} Train_Acc: {temp_acc:.4f}')
            else:
                print(f'Val loss: {temp_loss:.4f} Val_Acc: {temp_acc:.4f}')
    # Compute metrics
    total_loss /= (step + 1)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    current_time = time.time()
    if train is True:
        print(f'Train loss: {total_loss:.4f} Train_Acc: {accuracy:.4f} Training time: {(current_time-start_time)//60:.0f} minutes')
    else:
        print(f'Val_loss: {total_loss:.4f} Val_Acc: {accuracy:.4f} Validation time: {(current_time-start_time)//60:.0f} minutes')

    return precision,recall,f1,accuracy,total_loss


# Plot a line graph showing trend of losses (train, val) over time 
def plot_curve(train, val, param):
    plt.figure()
    steps = len(val)
    plt.plot(np.arange(1, steps+1,1),train[:steps], label='train '+param)
    plt.plot(np.arange(1,steps+1,1), val[:steps], label='validation '+param)
    plt.xticks(range(1,steps+1,2))
    plt.xlim(1,steps+1)
    plt.xlabel('epochs')
    plt.ylabel(param)
    #plt.title(param)
    plt.legend(loc='upper right')
    filename = param+'_profile.png'
    plt.savefig(filename)
    #plt.show()

