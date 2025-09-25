import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import Dataset ,DataLoader
from torchvision import datasets
from torch.utils.data import random_split
from tqdm import tqdm
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import time
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import StepLR
import io
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm  # Regular tqdm, not tqdm.auto
import os






class ModelFit():

    def __init__(self):
        self.model = None
        self.device = None
        self.writer = None
        self.scheduler = None  # Add this

        # Create writer
        
        
    
    def get_model(self,learning_rate , load_path = None ,loss_func = nn.CrossEntropyLoss(), device = 'cpu' , optim = None , model = None):


        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        if model:
            self.model = model
        if  self.model is None:
            raise ValueError("No model available. Either pass a model parameter or set self.model first")


        if load_path:
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))
      

        if optim:
            self.optimizer = optim
        elif  not optim:
            self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=learning_rate)


        self.model_name = self.model.__class__.__name__
        self.timestamp = time.strftime("%H_%M_S-%d-%M-%Y")
        self.log_dir = f"runs/real_data/{self.model_name}_{self.timestamp}"
        self.writer = SummaryWriter(log_dir=self.log_dir)

        
            
            

        return self.model , self.optimizer , loss_func



    def loss_batch(self,model, loss_func, xb, yb, opt=None):
        xb, yb = xb.to(self.device , non_blocking = True), yb.to(self.device, non_blocking = True)
        preds = model(xb)
        preds = preds.permute(0, 2, 3, 1)
        preds = preds.reshape(-1,9)
        yb=(yb - 1).reshape(-1)
        loss = loss_func(preds, yb)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), len(xb)
    
    
    
    def train_loss_over_batches(self,training_data,current_epoch,epoches,loss_func,opt):
        total_loss = 0
        count = 0
        self.model.train()

        progress_bar = tqdm(training_data, desc=f"Epoch {current_epoch+1}/{epoches}", leave=True)
        
        for xb, yb in progress_bar:
            xb, yb = xb.to(self.device), yb.to(self.device)
            loss, batch_size = self.loss_batch(self.model, loss_func, xb, yb, opt)
            total_loss += loss * batch_size
            count += batch_size
            progress_bar.set_postfix(train_loss=total_loss / count)
            train_loss = total_loss / count
            self.writer.add_scalar('Loss/train', train_loss, count)
        progress_bar.close()
        
        if self.scheduler is not None:
            self.scheduler.step()
        return train_loss

    def valid_loss_over_batches(self,validation_data,current_epoch,epoches,loss_func ):
        self.model.eval()
        val_losses = []
        val_nums = []

        with torch.no_grad():
            for xb, yb in validation_data:
                xb, yb = xb.to(self.device, non_blocking = True), yb.to(self.device, non_blocking = True)
                loss, batch_size = self.loss_batch(self.model, loss_func, xb, yb)
                val_losses.append(loss)
                val_nums.append(batch_size)

        val_loss = np.sum(np.multiply(val_losses, val_nums)) / np.sum(val_nums)
        self.writer.add_scalar('Loss/val', val_loss, current_epoch)

        return val_loss
    
    def model_training(self ,
                        epoches ,
                        loss_func ,
                        optimalizator , 
                        training_data , 
                        validation_data , 
                        weight_unfreeze_epoch,
                        scheduler_step_size , 
                        scheduler_gamma):
        
        self.model.to(self.device)
        self.scheduler = lr_scheduler.StepLR(optimalizator, step_size=scheduler_step_size, gamma=scheduler_gamma )  

        for epoch in range(epoches):
            if epoch == weight_unfreeze_epoch:
                for param in self.model.features.parameters():
                    param.requires_grad = True
            
            
            
            self.train_loss_over_batches(training_data=training_data,
                                         current_epoch=epoch,
                                         epoches=epoches,
                                         loss_func=loss_func,
                                         opt = optimalizator)
            self.valid_loss_over_batches(epoches = epoches , current_epoch= epoch , validation_data= validation_data , loss_func=loss_func )
        self.save_model()
        return self.model

    def save_model(self,filename="model.pt"):
        if self.model is None:
            raise ValueError("no model to save.")
        
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, filename)

        checkpoint = {
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': getattr(self, 'optimizer', None).state_dict() if hasattr(self, 'optimizer') else None
        }
        torch.save(checkpoint, path)
        print(f"Model saved: {path}")
    
    def model_testing(self,
                      test_loader,
                      loss_func,
                      class_names=None):
        self.model.eval()
        self.model.to(self.device)

        total_correct = 0
        total_samples = 0
        total_loss = 0

        all_preds = []
        all_labels = []
        progress_bar = tqdm(test_loader, desc="Testing", leave=False)
        with torch.no_grad():
            for xb, yb in progress_bar:
                xb, yb = xb.to(self.device), yb.to(self.device)
                outputs = self.model(xb)
                loss = loss_func(outputs, yb)

                total_loss += loss.item() * xb.size(0)

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == yb).sum().item()
                total_samples += xb.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        progress_bar.close()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        print(f"\nTest Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names if class_names else range(cm.shape[1]),
                    yticklabels=class_names if class_names else range(cm.shape[0]))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix\nLoss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        plt.tight_layout()
        """
        if self.writer:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            self.writer.add_figure("Confusion Matrix", buf, global_step=1)
            image = Image.open(buf)
            image = transforms.ToTensor()(image)
            self.writer.add_image("Confusion_Matrix", image ,1)
        plt.show()
        """
        if self.writer:
            self.writer.add_figure("Confusion Matrix", fig, global_step=1)

        plt.show()



        return avg_loss, accuracy, cm