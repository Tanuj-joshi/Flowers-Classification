import os
import argparse
import torch
from torchvision import transforms
from functions import predict, create_dataset, load_model
import sys, json
import imghdr
from PIL import Image
import matplotlib.pyplot as plt


def main(image_dir, arch, batchsize, device, trained_model):

    # Load any pretrained model if it already exists, otherwise load open-source model
    if os.path.exists(trained_model):
        model = load_model(arch,device,trained_model)
        loss_fn = torch.nn.NLLLoss()
    else:
        print("No pretrained model is present in the specified directory...")
        sys.exit()

    if os.path.isdir(image_dir):
        # Creating Dataset
        test_dataset = create_dataset(image_dir, train=False)
        # Prepare Dataloader
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
        # Save class_to_idx mapping to a JSON file
        # with open('class_to_idx.json', 'w') as f:
        #     json.dump(model.class_to_idx, f)
        print('Num of Test batches', len(testloader))
        # Prediction on Test data
        precision,recall,f1,accuracy,_= predict(model,None,testloader,loss_fn,device,train=False)
        print(f'Precision of the network on the test images: {precision:.4f}')
        print(f'Recall of the network on the test images: {recall:.4f}')
        print(f'F1 Score of the network on the test images: {f1:.4f}')
        print(f'Accuracy of the network on the test images: {accuracy:.4f}')

    elif imghdr.what(image_dir) is not None:
        # Scales, crops, and normalizes a PIL image for a PyTorch model,returns a Numpy array
        img_transform = transforms.Compose([transforms.Resize(255),
                                        transforms.RandomCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225])])

        image = Image.open(image_dir).convert('RGB')
        trans_image = img_transform(image)
        trans_image = trans_image.unsqueeze(0)  # Add batch dimension
        trans_image = trans_image.to(device)
        # using image, generate output
        with torch.no_grad():
            outputs = model(trans_image)
            _, predicted = torch.max(outputs, 1)
        # Get the class label (labels start from 0)
        class_id = predicted.item()
        # actual class labels start from 1 unlike above labels
        class_id+=1
        # Load class index to label mapping
        with open('cat_to_name.json') as f:
            class_idx = json.load(f)
        # find the flower class name
        class_label = class_idx[str(class_id)]
        # Display image
        plt.imshow(image)
        plt.title(f'Predicted: {class_label}')
        plt.axis('off')
        plt.show()
        #print(class_label)


if __name__ == "__main__":
    # Check if GPU is available
    if torch.cuda.is_available():
        print('Default GPU Device: {}'.format(torch.cuda.get_device_name(0)))
    else: 
        print('No GPU available, training on CPU; consider making n_epoch very small.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='', help='directory where training and validation images are located')
    parser.add_argument('--batch_size', type=int, help='batch size') 
    parser.add_argument('--model_path', type=str, help='trained model')
    parser.add_argument('--arch', type=str, help='base architecture: vgg13 or densenet121')
    parser.add_argument('--class_to_idx', type=str, help='json file containing flower class to id mapping')
    args = parser.parse_args()

    if (args.image_dir is None) or (args.model_path is None) or args.arch not in ['vgg13', 'densenet121']:
            print("--image_dir, --model_path, --arch must be either vgg13 or densenet121 ..... cannot be none")
            sys.exit()
    else:
        if (args.batch_size==None):
            main(args.image_dir, args.arch, 16, device, args.model_path)
        else:
            main(args.image_dir, args.arch, args.batch_size, device, args.model_path)


