import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image


class VGGSoundDatasetCLS(Dataset):
    def __init__(self, ds_path, num_cls, img_size, train_mode=True):
        super(VGGSoundDatasetCLS, self).__init__()
        self.train_mode = train_mode
        self.img_path = os.path.join(ds_path, "images")
        self._load_items(ds_path, num_cls)
        self.transform = transforms.Compose([transforms.Resize(img_size),
                                             transforms.RandomCrop(img_size),
                                             transforms.ToTensor()])
            
    def _load_items(self, ds_path, num_cls):
        mode = "train" if self.train_mode else "eval"
        
        with open(os.path.join(ds_path, f"meta_pseudo_{num_cls}_final_{mode}.txt"), "r") as metafile:
            all_lines = metafile.readlines()
        all_lines = [l.strip() for l in all_lines]
        self.items = [(l.split("|")[0], int(l.split("|")[1])) for l in all_lines]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.items[idx][0] + ".jpg")
        img = Image.open(img_path)
        img = self.transform(img)
        lbl = self.items[idx][1]
        return (img, lbl)

    def __len__(self):
        return len(self.items)   

def get_vggsoundcls_dataloader(config):
    dataset_train = VGGSoundDatasetCLS(config["ds_path"], config["num_cls"], config["img_size"], train_mode=True)
    dataloader_train = DataLoader(dataset_train, config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    
    dataset_eval = VGGSoundDatasetCLS(config["ds_path"], config["num_cls"], config["img_size"], train_mode=False)
    dataloader_eval = DataLoader(dataset_eval, config["batch_size"], shuffle=True, num_workers=config["num_workers"])

    return dataloader_train, dataloader_eval