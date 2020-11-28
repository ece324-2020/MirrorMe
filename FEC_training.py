from emb_dataset import TripletSet, split_data, getLoaders
import torch
from FECNet.models.FECNet import FECNet
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import copy

def save_model(model, path):
    try:
        torch.save(model.state_dict(), path)
        print("Model saved at {}".format(path))
    except:
        print("Model failed to save!")

def triplet_loss(out):
    e1 = out[0::3, :]
    e2 = out[1::3, :]
    e3 = out[2::3, :]

    d12 = (e1 - e2).pow(2).sum(1)
    d13 = (e1 - e3).pow(2).sum(1)
    d23 = (e2 - e3).pow(2).sum(1)

    alpha = 0.2

    d1 = F.relu((d12 - d13) + alpha)
    d2 = F.relu((d12 - d23) + alpha)

    d = torch.mean(d1 + d2)

    return d

def eval(out):
    with torch.no_grad():
        e1 = out[0::3, :]
        e2 = out[1::3, :]
        e3 = out[2::3, :]

        d12 = (e1 - e2).pow(2).sum(1)
        d13 = (e1 - e3).pow(2).sum(1)
        d23 = (e2 - e3).pow(2).sum(1)

        corr = (d12 < d13) * (d12 < d23)
        # print("Correct: %.1f" % (corr))

        n_corr = torch.sum(corr)

        return n_corr

def plot(x, y1, y2, mode, path):
    plt.clf()
    # y1 = signal.savgol_filter(y1, 21, 11, deriv=0)
    # y2 = signal.savgol_filter(y2, 21, 11, deriv=0)
    plt.plot(x, y1, x, y2)
    plt.xlabel('Epoch')
    plt.ylabel('{}'.format(mode))
    plt.title('{} per epoch'.format(mode))
    plt.legend(['Training {}'.format(mode), 'Validation {}'.format(mode)])
    plt.savefig(os.path.join(path,'{}.png'.format(mode)))
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--batch_size', type=int, default=64)
    parser.add_argument('-l','--lr', type=float, default=0.001)
    parser.add_argument('-n','--epochs', type=int, default=50)
    parser.add_argument('-w','--num_workers', type=int, default=6)
    

    args = parser.parse_args()
    
    seed = 0
    torch.manual_seed(seed)
    batch_size = args.batch_size
    num_workers = 6
    num_epochs = args.epochs

    image_PATH = '/home/MirrorMe/project/clean_dataset/train_data/'
    split_data('/home/MirrorMe/project/labels/output.csv',0.2)
    trainloader, validloader = getLoaders(image_PATH,'/home/MirrorMe/project/labels/train.csv','/home/MirrorMe/project/labels/validation.csv',224,batch_size=batch_size,num_workers=num_workers)

    saved_models_dir = 'saved_models'
    plots_dir = 'plots'
    save_freq = 5

    if (not os.path.exists(saved_models_dir)):
        os.mkdir(saved_models_dir)

    if (not os.path.exists(plots_dir)):
        os.mkdir(plots_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = FECNet(pretrained=True)

    #* If we want to load a previous checkpoint, use this:
    # checkpoint_file = 'model_epoch_{}'.format(last)
    # checkpoint = torch.load(os.path.join(saved_models_dir, checkpoint_file) + ".pkl")
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainAccuracies = np.array([])
    trainlosses = np.array([])
    validAccuracies = np.array([])
    validlosses = np.array([])
    
    best_validAcc = 0.0
    start = time.time()
    for epoch in range(num_epochs):
        print('\n-----------------------------------------------')
        print("Epoch #: {}\n".format(epoch+1))

        for phase in ['train','val']:
            with torch.set_grad_enabled(phase == 'train'):
                if phase == 'train':
                    model = model.train()

                    running_loss = 0.0
                    corrects = 0

                    for i, batch in enumerate(trainloader,0):
                        optimizer.zero_grad()
                        batch = batch.to(device)
                        out = model(batch.view(-1, 3, 224, 224))
                        # print("OUT shape: {}".format(out.shape))

                        batch_loss = triplet_loss(out)
                        
                        cor = eval(out)

                        corrects += cor.item()
                        batch_loss.backward()
                        running_loss += batch_loss.item()
                        optimizer.step()

                    print("Training Loss: ", f'{(running_loss/(i+1)):.4f}')
                    trainlosses = np.append(trainlosses, (running_loss/(i+1)))
                    trainAcc = float(corrects)/len(trainloader.dataset)
                    print("Training Accuracy: ", f'{trainAcc:.4f}')
                    trainAccuracies = np.append(trainAccuracies, trainAcc)

                else:
                    model = model.eval()
                    validloss = 0.0
                    corrects = 0

                    for i, batch in enumerate(validloader,0):
                        batch = batch.to(device)
                        out = model(batch.view(-1, 3, 224, 224))
                        batch_loss = triplet_loss(out)
                        
                        cor = eval(out)
                        corrects += cor.item()
                        validloss += batch_loss.item()
                    validloss = validloss/(i+1)
                    validAcc = float(corrects)/len(validloader.dataset)
                    print("Validation Loss: ", f'{validloss:.4f}')
                    print("Validation Accuracy: ", f'{validAcc:.4f}')
                    validlosses = np.append(validlosses, validloss)
                    validAccuracies = np.append(validAccuracies, validAcc)

        #? Update best validation accuracy
        if best_validAcc < validAccuracies[-1]:
            best_validAcc = validAccuracies[-1]
            save_file = 'model_epoch_{}'.format(epoch)
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': trainlosses[-1]
            }, os.path.join(saved_models_dir, save_file) + ".pkl")
            print("SAVED CHECKPOINT")
        # #Save model
        # if (epoch % save_freq == 0):
        #     torch.save({
                
        #     })
        #     save_file = 'model_epoch_{}'.format(epoch)
        #     save_model(model, os.path.join(saved_models_dir, save_file))
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    plot(range(num_epochs), trainlosses, validlosses,'Loss',plots_dir)
    plot(range(num_epochs), trainAccuracies, validAccuracies, 'Accuracy',plots_dir)

    model.load_state_dict(best_model_wts)
    model = model.cpu()
    model_file = 'final_FEC.pt'
    torch.save(model.state_dict(), os.path.join(saved_models_dir, model_file))
    print("SAVED Final!")

if __name__ == '__main__':
    main()