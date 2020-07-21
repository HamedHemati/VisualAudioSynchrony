import os
import torch
from vasync.data.vggsound_dataloader import get_vggsoundcls_dataloader
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


class ClassficationTrainer():
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self._load_data()
        self._create_dirs()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), self.config["lr"]) 

    def _load_data(self):
        self.dataloader = get_vggsoundcls_dataloader(self.config)
    
    def _create_dirs(self):
        self.checkpoint_path = os.path.join(self.config["outputs_path"], self.config["pretext_task"], self.config["run_name"])
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def _save_fig_path(self, X, path):
        fig = plt.figure()
        plt.plot(X)
        fig.savefig(path)

    def _save_fig_path_global(self, X, Y, path):
        fig = plt.figure()
        plt.plot(X, Y)
        fig.savefig(path)

    def train(self):
        epochs = self.config["epochs"]
        global_itr = 0
        loss_global_itr_list = []
        loss_global_itr_list_x = []
        loss_epochs_list =[]
        for epoch in range(epochs):
            print(f"Epoch {epoch}:\n")
            avg_loss_epoch = 0.0
            for itr, (img, lbl) in enumerate(self.dataloader):
                img, lbl = img.to(self.device), lbl.to(self.device)
                self.model.zero_grad()
                out = self.model(img)
                loss = self.criterion(out, lbl)
                loss.backward()
                self.optimizer.step()
                if itr % 50 == 0 :
                    print(f"Epoch {epoch}: {itr}/{len(self.dataloader)} - Loss: {loss.item()}")
                    loss_global_itr_list.append(loss.item())
                    loss_global_itr_list_x.append(global_itr)
                    itr_plot_path = os.path.join(self.checkpoint_path, "loss_itr.png")
                    self._save_fig_path_global(loss_global_itr_list_x, loss_global_itr_list, itr_plot_path)
                avg_loss_epoch += loss.item()
                global_itr += 1

            avg_loss_epoch = avg_loss_epoch / len(self.dataloader)
            loss_epochs_list.append(avg_loss_epoch)
            avg_plot_path = os.path.join(self.checkpoint_path, "loss_epoch.png")
            self._save_fig_path(loss_epochs_list, avg_plot_path)

            print(f"Saving checkpoint for epoch {epoch}\n\n")
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, f"checkpoint_epoch_{epoch}.pt"))


    def eval(self):
        pass