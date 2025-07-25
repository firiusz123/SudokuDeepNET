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







class ModelFit():

    def __init__(self):
        self.model = None
        self.device = None
        self.writer = None
        self.scheduler = None  # Add this

        # Create writer
        
        
    
    def get_model(self,learning_rate , load_path = None ,loss_func = nn.CrossEntropyLoss(), device = 'cpu' , optim = None , model = None):


        if device == 'cpu':
            self.device=torch.device('cpu')
        else:
             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
        if model:
            self.model = model
        if  self.model is None:
            raise ValueError("No model available. Either pass a model parameter or set self.model first")


        if load_path:
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))
      

        if optim:
            optimizer = optim
        elif  not optim:
            optimizer  = torch.optim.Adam(self.model.parameters(), lr=learning_rate)


        self.model_name = self.model.__class__.__name__
        self.timestamp = time.strftime("%H_%M_S-%d-%M-%Y")
        self.log_dir = f"runs/real_data/{self.model_name}_{self.timestamp}"
        self.writer = SummaryWriter(log_dir=self.log_dir)

        
            
            

        return self.model , optimizer , loss_func



    def loss_batch(self,model, loss_func, xb, yb, opt=None):
        xb, yb = xb.to(self.device), yb.to(self.device)
        preds = model(xb)
        loss = loss_func(preds, yb)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), len(xb)
    
    
    
    def train_loss_over_batches(self,training_data,current_epoch,epoches,loss_func,opt,writer):
        total_loss = 0
        count = 0
        self.model.train()

        progress_bar = tqdm(training_data, desc=f"Epoch {current_epoch+1}/{epoches}", leave=False)
        
        for xb, yb in progress_bar:
            xb, yb = xb.to(self.device), yb.to(self.device)
            loss, batch_size = self.loss_batch(self.model, loss_func, xb, yb, opt)
            total_loss += loss * batch_size
            count += batch_size
            progress_bar.set_postfix(train_loss=total_loss / count)
            train_loss = total_loss / count
            writer.add_scalar('Loss/train', train_loss, count)
        
        if self.scheduler is not None:
            self.scheduler.step()
        return train_loss

    def valid_loss_over_batches(self,validation_data,current_epoch,epoches,loss_func,writer ):
        self.model.eval()
        val_losses = []
        val_nums = []

        with torch.no_grad():
            for xb, yb in validation_data:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss, batch_size = self.loss_batch(self.model, loss_func, xb, yb)
                val_losses.append(loss)
                val_nums.append(batch_size)

        val_loss = np.sum(np.multiply(val_losses, val_nums)) / np.sum(val_nums)
        writer.add_scalar('Loss/val', val_loss, current_epoch)

        return val_loss
    
    def model_training(self ,
                        epoches ,
                        loss_func ,
                        optimalizator , 
                        training_data , 
                        validation_data , 
                        weight_unfreeze_epoch,
                        scheduler_step_size , 
                        scheduler_gamma ,
                          writer):
        
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
                                         opt = optimalizator,
                                         writer = writer)
            self.valid_loss_over_batches(epoches = epoches , current_epoch= epoch , validation_data= validation_data , loss_func=loss_func , writer = writer)
        return self.model

    def model_testing(self,
                      
                      test_loader,
                      loss_func,
                      class_names=None,
                      writer = None):
        self.model.eval()
        self.model.to(self.device)

        total_correct = 0
        total_samples = 0
        total_loss = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in tqdm(test_loader, desc="Testing", leave=False):
                xb, yb = xb.to(self.device), yb.to(self.device)
                outputs = self.model(xb)
                loss = loss_func(outputs, yb)

                total_loss += loss.item() * xb.size(0)

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == yb).sum().item()
                total_samples += xb.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        print(f"\nTest Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names if class_names else range(cm.shape[1]),
                    yticklabels=class_names if class_names else range(cm.shape[0]))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix\nLoss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        plt.tight_layout()
        
        if writer:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)
            image = transforms.ToTensor()(image)
            writer.add_image("Confusion_Matrix", image ,1)
        plt.show()

        return avg_loss, accuracy, cm

        



    def fit(self,epoches,loss_function,optimalizator,training_data,validation_data,weightt_unfreeze_epoch,writer):
        pass
    
















class SimpleSudokuCNN(nn.Module):
        def __init__(self):
            super(SimpleSudokuCNN, self).__init__()
            # wejście: (batch, 10, 9, 9)
            self.conv1 = nn.Conv2d(10, 32, kernel_size=3, padding=1)  # (batch, 32, 9, 9)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (batch, 64, 9, 9)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # (batch, 128, 9, 9)
            
            self.dropout = nn.Dropout(0.3)
            
            # ostatnia warstwa konwolucyjna zmienia liczbę kanałów na 9 (klasy od 0 do 8)
            self.conv_out = nn.Conv2d(128, 9, kernel_size=1)           # (batch, 9, 9, 9)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.dropout(x)
            x = self.conv_out(x)  # logits dla każdej klasy na każdej pozycji
            # output shape: (batch, 9, 9, 9)
            # Jeśli potrzebujesz (batch, 9, 9, 9), permutuj: x.permute(0, 2, 3, 1)
            return x





# losowe dane: batch 100, 10 kanałów, 9x9 sudoku one-hot
x_dummy = torch.randn(100, 10, 9, 9)
# losowe etykiety: batch 100, 9x9, klasa od 0 do 8
y_dummy = torch.randint(0, 9, (100, 9, 9))

dataset = TensorDataset(x_dummy, y_dummy)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset, batch_size=16)



train_dataset = TensorDataset(x_dummy, y_dummy)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=64)  # użyj tego samego jako walidacji

# Model i trenowanie
# Define model first
model = SimpleSudokuCNN()

# Create trainer and get components
trainer = ModelFit()
model, optimizer, loss_func = trainer.get_model(
    learning_rate=1e-3, 
    model=model
)

# Train (using the returned optimizer and loss_func)
trainer.model_training(
    epoches=10,
    loss_func=loss_func,
    optimalizator=optimizer,  # Use the one returned by get_model
    training_data=train_loader,
    validation_data=val_loader,
    weight_unfreeze_epoch=None,
    scheduler_step_size=5,
    scheduler_gamma=0.5,
    writer=trainer.writer
)

