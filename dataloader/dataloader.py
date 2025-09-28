import torch
from torch.utils.data import Dataset ,DataLoader

class SudokuDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.puzzles = data['puzzles']     
        self.solutions = data['solutions'] 
        self.hole_count = data['hole_counts']
        self.id = data['ids']

        # Walidacja
        assert isinstance(self.puzzles, torch.Tensor)
        assert isinstance(self.solutions, torch.Tensor)
        assert self.puzzles.shape == self.solutions.shape, (
            f"Shapes mismatch: puzzles {self.puzzles.shape}, solutions {self.solutions.shape}"
        )

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        x = self.puzzles[idx]     
        y = self.solutions[idx] - 1   
        x = x.unsqueeze(0).float()
        y = y.long() 

        return x, y
