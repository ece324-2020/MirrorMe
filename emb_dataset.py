import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from FECNet.models.FECNet import FECNet


class TripletSet(Dataset):
    def __init__(self,label_csv,dataset_path,img_size):
        """

        Args:
            label_csv (string): Path to the csv file containing the labels for triplets
            dataset_path (string): Path to the directory containing the images described in label_csv
            img_size (int): Image size after rescaling in transforms
        """    
        super(TripletSet,self).__init__()
        self.data = pd.read_csv(label_csv)
        self.PATH = dataset_path
        self.size = img_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        tf = transforms.Compose([
            transforms.Lambda(lambda x:Image.open(x).convert("RGB")),
            transforms.ToTensor(),
            transforms.Resize((self.size,self.size))
        ])

        # ?Get triplet and label from csv dataframe
        triplet = self.data.iloc[idx,0:3].tolist()
        label = self.data.iloc[idx,3].tolist()

        #? Reorder images such that the odd one is the 3rd image
        if label == 1:
            order = [1,2,0]
        elif label == 2:
            order = [0,2,1]
        elif label == 3:
            order = [0,1,2]
        else:
            print("LABEL: {}".format(label))
            raise ValueError("INVALID label")
        triplet = [triplet[i] for i in order]

        #? Add path to the images
        triplet = [self.PATH+img for img in triplet]
        #? Pass in triplet for transformations
        for i in range(len(triplet)):
            try:
                triplet[i] = tf(triplet[i])
            except Exception as e:
                print(e)
                continue
        triplet = torch.stack(triplet, dim=0)   # dim: (3,3,224,224)

        return triplet

def split_data(label_csv, valid_ratio = 0.2):
    """Split main csv file into training and validation csv files

    Args:
        label_csv (string): Path to the csv file containing the labels for triplets
        valid_ratio (float, optional): ratio to use for validation set. Defaults to 0.2.

    Returns:
        [boolean]: True
    """    
    #? Import data.tsv as dataframe
    data = pd.read_csv(label_csv,sep=',',index_col=False)
    X = data.iloc[:,0:3]
    y = data.iloc[:,3]

    #? Split train and test
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_ratio, random_state=0)

    #? Put the input and label together

    training = pd.concat([X_train, y_train], axis=1)
    valid = pd.concat([X_valid, y_valid], axis=1)

    #? Save datasets
    try:
        training.to_csv('/home/MirrorMe/project/labels/train.csv', index=False)
        valid.to_csv('/home/MirrorMe/project/labels/validation.csv', index=False)
    except Exception as e:
        print(e)
    else:
        print("Datasets successfully split!")
    finally:
        return True
    
def getLoaders(dataset_path, trainCSV, validCSV, img_size, batch_size, num_workers=0):
    """Generates the dataloaders for training and validation

    Args:
        dataset_path (string): Path to the directory containing the images described in label_csv
        trainCSV (string): Path to the training csv file
        validCSV (string): Path to the validation csv file
        img_size (int): Image size (check dataset class for more info)
        batch_size (int): batch size to be used for loader
        num_workers (int, optional): num_workers for dataloader. Defaults to 0.

    Returns:
        [DataLoader]: trainLoader, validLoader
    """
    #? Instantiate the datasets
    trainset = TripletSet(trainCSV,dataset_path,img_size)
    validset = TripletSet(validCSV,dataset_path,img_size)

    #? Get the loaders
    trainLoader = DataLoader(trainset, batch_size=batch_size,shuffle=True,num_workers=num_workers)
    validLoader = DataLoader(validset, batch_size=batch_size,shuffle=True,num_workers=num_workers)

    return trainLoader, validLoader

def main():
    seed = 0
    torch.manual_seed(seed)

    image_PATH = '/home/MirrorMe/project/clean_dataset/train_data/'
    split_data('/home/MirrorMe/project/labels/output.csv',0.2)
    trainloader, validloader = getLoaders(image_PATH,'/home/MirrorMe/project/labels/train.csv','/home/MirrorMe/project/labels/validation.csv',224,batch_size=1,num_workers=0)
    dataiter1 = iter(trainloader)
    x1 = dataiter1.next()
    dataiter2 = iter(validloader)
    x2 = dataiter2.next()
    print(x1.shape)
    print(x2.shape)

    model = FECNet(pretrained=True)

    out = model(x1.view(-1, 3, 224, 224).cuda())

    print(out)

if __name__ == '__main__':
    main()

