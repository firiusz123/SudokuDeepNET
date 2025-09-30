import torch
from torch.utils.data import DataLoader, random_split
from dataloader import SudokuDataset
from model import SudokuNet
from pipeline import ModelFit




def main():

        full_dataset = SudokuDataset("DataGeneration/data/data2.pt")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
        print(f"Using device: {device}")
        model = SudokuNet().to(device)


        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])
        workers_num =5
        # Loaders
        pin_memory_state = False
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,pin_memory=pin_memory_state, num_workers=workers_num)
        val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False,pin_memory=pin_memory_state, num_workers=workers_num)
        test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False,pin_memory=pin_memory_state, num_workers=workers_num)




        model = SudokuNet()

        # Create trainer and get components
        trainer = ModelFit()
        model, optimizer, loss_func = trainer.get_model(
            learning_rate=1e-3, 
            model=model,load_path="runs/real_data/SudokuNet_20_04_S-30-04-2025/model.pt"
        )
        # Train (using the returned optimizer and loss_func)
        
        trainer.model_testing(test_loader,loss_func,class_names=9)
        
if __name__ == '__main__':
        main()
