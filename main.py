import random
import torch
import time  # Import the time module

class SudokuGenerator:
    def __init__(self):
        self.base = 3  # Size of the base Sudoku grid (for a 9x9 grid)
        self.side = self.base * self.base
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use CUDA if available

    # Create a full solution using a backtracking algorithm
    def solve(self, grid):
        find = self.find_empty(grid)
        if not find:
            return True  # Solved
        row, col = find

        for num in range(1, self.side + 1):
            if self.is_safe(grid, num, row, col):
                grid[row][col] = num

                if self.solve(grid):
                    return True

                grid[row][col] = 0

        return False

    # Check if it's safe to place a number in a given cell
    def is_safe(self, grid, num, row, col):
        if num in grid[row]:
            return False
        if num in [grid[i][col] for i in range(self.side)]:
            return False
        box_x = row // self.base * self.base
        box_y = col // self.base * self.base
        for i in range(self.base):
            for j in range(self.base):
                if grid[box_x + i][box_y + j] == num:
                    return False
        return True

    # Find the next empty cell
    def find_empty(self, grid):
        for i in range(self.side):
            for j in range(self.side):
                if grid[i][j] == 0:
                    return (i, j)
        return None

    # Generate a fully solved grid
    def generate_solution_grid(self):
        grid = torch.zeros((self.side, self.side), dtype=torch.int8, device=self.device)  # Use CUDA tensor
        self.fill_diagonal_boxes(grid)
        self.solve(grid)
        return grid

    # Fill the diagonal 3x3 boxes with random numbers
    def fill_diagonal_boxes(self, grid):
        for i in range(0, self.side, self.base):
            self.fill_box(grid, i, i)

    # Fill a 3x3 box
    def fill_box(self, grid, row, col):
        numbers = list(range(1, self.side + 1))
        random.shuffle(numbers)
        for i in range(self.base):
            for j in range(self.base):
                grid[row + i][col + j] = numbers.pop()

    # Remove numbers to create the puzzle
    def remove_numbers(self, grid, num_holes):
        puzzle = grid.clone()  # Create a copy of the grid
        count = num_holes
        while count > 0:
            row = random.randint(0, self.side - 1)
            col = random.randint(0, self.side - 1)
            if puzzle[row][col] != 0:
                puzzle[row][col] = 0
                count -= 1
        return puzzle

    # Generate multiple puzzles and solutions as int8 tensors
    def generate_puzzles(self, num_puzzles=5):
        puzzles = torch.zeros((num_puzzles, self.side, self.side), dtype=torch.int8, device=self.device)
        solutions = torch.zeros((num_puzzles, self.side, self.side), dtype=torch.int8, device=self.device)

        for idx in range(num_puzzles):
            start_time = time.time()  # Record the start time for puzzle generation

            solution = self.generate_solution_grid()
            num_holes = random.randint(12, 40)  # Random number of holes between 12 and 40
            puzzle = self.remove_numbers(solution, num_holes)
            puzzles[idx] = puzzle
            solutions[idx] = solution
            
            end_time = time.time()  # Record the end time for puzzle generation
            elapsed_time = end_time - start_time  # Calculate elapsed time
            print(f"Puzzle {idx + 1} generated in {elapsed_time:.4f} seconds.")  # Print the time taken for each puzzle
        
        return puzzles, solutions

# Example usage
sudoku = SudokuGenerator()
puzzles, solutions = sudoku.generate_puzzles(num_puzzles=30)

print(puzzles.size(), solutions.size())
