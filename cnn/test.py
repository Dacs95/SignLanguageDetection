import torch
import torchvision
import torchvision.transforms as transforms

#Transformar a tensor
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root='./../DataSet', transform=transform)

classes = ('a','b','c','d','e','f','g','h','i','l','m','n','o','p','r','s','t','u','v','w','y')