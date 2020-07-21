import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image


class VGGSoundDatasetCLS(Dataset):
    def __init__(self, ds_path, num_cls):
        super(VGGSoundDatasetCLS, self).__init__()
        self.img_path = os.path.join(ds_path, "images")
        self._load_items(ds_path, num_cls)
        self.transform = transforms.Compose([transforms.Resize(250),
                                             transforms.RandomCrop(245),
                                             transforms.ToTensor()])

    def _load_items(self, ds_path, num_cls):
        with open(os.path.join(ds_path, f"meta_pseudo_{num_cls}_final.txt"), "r") as metafile:
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
    dataset = VGGSoundDatasetCLS(config["ds_path"], config["num_cls"])
    dataloader = DataLoader(dataset, config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    return dataloader