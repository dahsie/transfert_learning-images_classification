import torch
import torch.nn as nn

import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def get_model_file_size(model=None,model_path=None):
    """Function to get the model size. It will be apply torch model load from torchvision or to torch saved model through the path
    Args:
        model : torch model to get the size
        model_path : 
    """

    if model: 
        temp_file = 'temp_model.pth'
        torch.save(model.state_dict(), temp_file)
        file_size = os.path.getsize(temp_file)/(1024*1024)

        os.remove(temp_file)
        print(f" models size {file_size} \n")
        return file_size
    else:
        file_size = os.path.getsize(model_path)/(1024*1024)
        print(f" models size {file_size} \n")
        return file_size
    
def load_data(root,transform=None):
        """Function to load training or evaluation dataset
        Args:
            root: path to the data to be be loaded
            transform : prepeocessing to apply to the loaded data
        Retrun a dataset which we want to load
        """
        return torchvision.datasets.ImageFolder(root=root, transform=transform)


def make_prediction(model,test_loader,device="cuda"):
    """Function de make prediction on a trainrd model
    Args:
        model : model that we want to evaluate
        test_loader: test dataset 
        device : "cud" or "cpu"
    Return  model accuracy
    """
    model.eval() # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():

        correct = 0
        total = 0
        
        for images, labels in tqdm(test_loader):

            images = images.to(device)
            labels = labels.to(device)
            # start=time()
            outputs = model(images)
            # end=time()
            # print((end-start)*1000)
        

        
            _, predicted = torch.max(outputs.data, 1)
           
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        accuracy=100 * correct / total
    print('Test Accuracy of the model : {} %'.format(accuracy))
    return accuracy


def fune_tune_model(model,num_class=10):
    """
    This function will adapt the pretrained model for our use case. The output of our model will be 10 instead of 1000.

    Args:
        model (torch.nn.Module): Pretrained model to adapt.
        num_class (int): The number of classes to predict (10 in our case).

    Returns:
        torch.nn.Module: The modified model.
    """

    with torch.no_grad(): 
        input_features=model.heads.head.in_features
        model.heads.head=nn.Linear(in_features=input_features,out_features=num_class,bias=True)

    return model
    with torch.no_grad(): 
        input_features=model.heads.head.in_features
        model.heads.head=nn.Linear(in_features=input_features,out_features=num_class,bias=True)

    return model

def training_torch_model(model,trainloader, epochs,criterion,optimizer,device,save_path):
    """Training an adatative model on imagenet dataset
    Args:
        model: model to be train
        trainloader : Torch DataLoader which contain the training dataset
        epochs: number of iteration on which the model will be trained
        criterion : loss function according on which the model will be trained
        optimizer : the optimization algorithm to use during training ("adam","sgd",...)
        device : cuda or cpu
        save_path : path where the training model will be saved afin the training has been ended
    """

    model.train().to(device)

    print("start training ...")

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        n=len(trainloader)
        i=0
        #accuracy=make_prediction(model,testloader,device)
        for images,labels in tqdm(trainloader):
            
            i=i+1
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            images=images.to(device)
            labels=labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000}')
                j=j+10
                running_loss = 0.0
    torch.save(model,save_path)

    print('Finished Training')


def main():
   
    torch.manual_seed(0)
    weight=ViT_B_16_Weights.IMAGENET1K_V1

    preprocess=weight.transforms()
    dataset=load_data("/home/dah/dataset/train",preprocess)

    batch_size=25

    train_loader=DataLoader(dataset=dataset, batch_size=batch_size,shuffle=True,num_workers=15)

    epochs=10
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" device {device}")


    model_path='/home/dah/computer_vision/models/vit_b_16'
    
    model=torch.load(model_path)

    #model evalution before training

    

    #accuracy=make_prediction(model,testloader,device)


    #Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
    save_trained_model_path='/home/dah/computer_vision/models/vit_b_16_v2'
    training_torch_model(model,train_loader,epochs,criterion,optimizer,device,save_trained_model_path)
    
    #Evaluation du model

    #del model

    #Model evalution after training

    model=torch.load(save_trained_model_path).to(device)
    result=get_model_file_size(model)

    testdataset=load_data("/home/dah/dataset/val",preprocess)
    testloader=DataLoader(testdataset,batch_size=250,shuffle=False, num_workers=15)

    accuracy=make_prediction(model,testloader,device)





if __name__=='__main__':
    main()


