import torch
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader

class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def load_dataset(root, name, train_percent, mean=(0,0,0), std=(1,1,1)):
    transformations = transforms.Compose([transforms.CenterCrop(224),transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                    transforms.ToTensor(),transforms.Normalize(mean,std)])
    dataset = torchvision.datasets.ImageFolder(os.path.join(root,name),transform=transformations)
    classes_dict = {i:data for i,data in enumerate(dataset.classes)}
    train_length = int(len(dataset) * train_percent)
    test_length = len(dataset) - train_length
    train, test = torch.utils.data.random_split(dataset,[train_length,test_length])
    return train,test, classes_dict