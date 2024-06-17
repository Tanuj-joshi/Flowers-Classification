import os, sys
import argparse
import torch
from torchsummary import summary
from functions import predict, create_dataset, plot_curve, load_model
from os import makedirs
import time
# Set CUDA_LAUNCH_BLOCKING=1 to help with debugging CUDA errors
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main(image_dir, arch, epoch, batchsize, device, trained_model):

    #generate dataset
    train_dataset, valid_dataset = create_dataset(image_dir, train=True)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batchsize, shuffle=False)
    # Load any pretrained model if it already exists, otherwise load open-source model
    model = load_model(arch, device, trained_model)
    summary(model, (3, 224, 224))
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)

    print("Start training model")
    start_time = time.time()
    running_train_loss = []
    running_train_acc = [0]
    running_train_precision = []
    running_train_recall = []
    running_val_loss = []
    running_val_acc = [0]
    running_val_precision = []
    running_val_recall = []

    for e in range(epoch):
        print('='*20)
        print(f'Starting epoch {e + 1}/{epoch}')
        print('='*20)

        train_precision,train_recall,train_f1,train_acc,train_loss = predict(model,optimizer,trainloader,loss_fn,device,train=True)
        val_precision,val_recall,val_f1,val_acc,val_loss = predict(model,optimizer,validloader,loss_fn,device,train=False)
        current_time = time.time()
        print(f'Epoch: {e+1}/{epoch} Train_loss: {train_loss:.4f} Train_Acc: {train_acc:.4f} Val_loss: {val_loss:.4f}, Val_Acc: {val_acc:.4f} Training_time: {(current_time-start_time)//60:.0f} minutes')
        running_train_loss.append(train_loss)
        running_train_acc.append(train_acc)
        running_train_precision.append(train_precision)
        running_train_recall.append(train_recall)
        running_val_loss.append(val_loss)
        running_val_acc.append(val_acc)
        running_val_precision.append(val_precision)
        running_val_recall.append(val_recall)

        # path where we want to save the trained model
        makedirs(trained_model, exist_ok=True)

        model.train()
        if val_acc > 0.95:
            print('Performance condition satisfied, Stopping..')
            save_file = trained_model +'epoch'+str(e+1)+'_best_classifier.pt'
            torch.save(model.state_dict(), save_file)
            return
        plot_curve(running_train_loss, running_val_loss, param='loss')
        plot_curve(running_train_acc, running_val_acc, param='acc')
        plot_curve(running_train_precision, running_val_precision, param='precision')
        plot_curve(running_train_recall, running_val_recall, param='recall')
        
    print("Training Complete...")
    save_file = trained_model +'epoch'+str(e+1)+'_classifier.pt'
    torch.save(model.state_dict(), save_file)


if __name__ == "__main__":

    # Check if GPU is available
    if torch.cuda.is_available():
        print('Default GPU Device: {}'.format(torch.cuda.get_device_name(0)))
    else: 
        print('No GPU available, training on CPU; consider making n_epoch very small.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='', help='directory where training and validation images are located')
    parser.add_argument('--epoch', type=int, help='total training epoch')
    parser.add_argument('--batch_size', type=int, help='batch size') 
    parser.add_argument('--model_path', type=str, help='directory where trained model should be saved')
    parser.add_argument('--arch', type=str, help='base architecture: vgg13 or densenet121')
    args = parser.parse_args()

    if (args.image_dir is None) or (args.model_path is None) or args.arch not in ['vgg13', 'densenet121']:
            print("--image_dir, --model_path, --arch must be either vgg13 or densenet121 ..... cannot be none")
            sys.exit()
    else:
        if (args.batch_size==None):
            main(args.image_dir, args.arch, args.epoch, 16, device, args.model_path)
        else:
            main(args.image_dir, args.arch, args.epoch, args.batch_size, device, args.model_path)