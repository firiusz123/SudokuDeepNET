import os
import random
import time
import argparse
import torch
import h5py
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError

class SudokuGenerator:
	def __init__(self):
		self.base = 3
		self.side = self.base * self.base

	def solve(self, grid, row, col, num):
		for n in range(1, self.side + 1):
			if self.is_safe(grid, n, row, col):
				grid[row][col] = n
				next_row, next_col = self.find_empty(grid)
				if next_row is None:
					return True
				if self.solve(grid, next_row, next_col, None):
					return True
				grid[row][col] = 0
		return False

	def is_safe(self, grid, num, row, col):
		return (num not in grid[row] and
				num not in grid[:, col] and
				num not in grid[row // self.base * self.base: (row // self.base + 1) * self.base,
							   col // self.base * self.base: (col // self.base + 1) * self.base])

	def find_empty(self, grid):
		for i in range(self.side):
			for j in range(self.side):
				if grid[i][j] == 0:
					return i, j
		return None, None

	def generate_solution_grid(self):
		grid = torch.zeros((self.side, self.side), dtype=torch.int8)
		self.fill_all_boxes(grid)
		self.solve(grid, 0, 0, None)
		return grid

	def fill_all_boxes(self, grid):
		for i in range(self.base):
			for j in range(self.base):
				self.fill_box(grid, i * self.base, j * self.base)

	def fill_box(self, grid, row, col):
		numbers = list(range(1, self.side + 1))
		random.shuffle(numbers)
		for i in range(self.base):
			for j in range(self.base):
				grid[row + i][col + j] = numbers.pop()

	def remove_numbers(self, grid, num_holes):
		puzzle = grid.clone()
		cells = list(range(self.side * self.side))
		random.shuffle(cells)
		for index in cells[:num_holes]:
			row, col = divmod(index, self.side)
			puzzle[row][col] = 0
		return puzzle

	def generate_single_puzzle(self, idx):
		solution = self.generate_solution_grid()
		num_holes = random.randint(12, 48)
		puzzle = self.remove_numbers(solution, num_holes)
		return puzzle, solution, num_holes, idx

	def generate_puzzles(self, num_puzzles=5, max_workers=20):
		puzzles = torch.zeros((num_puzzles, self.side, self.side), dtype=torch.int8)
		solutions = torch.zeros((num_puzzles, self.side, self.side), dtype=torch.int8)
		hole_counts = torch.zeros(num_puzzles, dtype=torch.int32)
		ids = torch.arange(num_puzzles, dtype=torch.int32)

		with ThreadPoolExecutor(max_workers=max_workers) as executor:
			futures = [executor.submit(self.generate_single_puzzle, i) for i in range(num_puzzles)]
			for idx, future in tqdm(enumerate(futures), total=num_puzzles, desc="Generating Puzzles"):
				try:
					puzzle, solution, holes, pid = future.result()
					puzzles[idx] = puzzle
					solutions[idx] = solution
					hole_counts[idx] = holes
				except TimeoutError:
					print(f"Puzzle {idx + 1} timed out.")

		return puzzles, solutions, hole_counts, ids

	def change_to_one_hot(self, data):
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		n_samples = data.shape[0]
		one_hot_data = torch.zeros(n_samples, 9, 9, 10, device=device)

		for n in range(n_samples):
			values = data[n].to(torch.long)
			for i in range(9):
				for j in range(9):
					v = values[i][j]
					if v > 0:
						one_hot_data[n][i][j][v] = 1
		return one_hot_data.cpu()

	def save_dataset(self, puzzles, solutions, hole_counts, ids, path='sudoku_dataset.pt', one_hot=False):
		os.makedirs(os.path.dirname(path), exist_ok=True)

		if one_hot:
			puzzles = self.change_to_one_hot(puzzles)
			solutions = self.change_to_one_hot(solutions)

		if path.endswith('.pt'):
			torch.save({
				'puzzles': puzzles,
				'solutions': solutions,
				'hole_counts': hole_counts,
				'ids': ids
			}, path)
			print("Saved as PyTorch file:", path)

		elif path.endswith('.h5'):
			with h5py.File(path, 'w') as hf:
				hf.create_dataset('puzzles', data=puzzles.numpy(), chunks=True, compression="gzip")
				hf.create_dataset('solutions', data=solutions.numpy(), chunks=True, compression="gzip")
				hf.create_dataset('hole_counts', data=hole_counts.numpy())
				hf.create_dataset('ids', data=ids.numpy())
			print("Saved as HDF5 file:", path)

		else:
			raise ValueError("Unsupported file format. Use .pt or .h5")

	def load_dataset(self, path='sudoku_dataset.pt'):
		if path.endswith('.pt'):
			data = torch.load(path)
			return data['puzzles'], data['solutions'], data['hole_counts'], data['ids']
		elif path.endswith('.h5'):
			with h5py.File(path, 'r') as hf:
				puzzles = torch.tensor(hf['puzzles'][:])
				solutions = torch.tensor(hf['solutions'][:])
				hole_counts = torch.tensor(hf['hole_counts'][:])
				ids = torch.tensor(hf['ids'][:])
			return puzzles, solutions, hole_counts, ids
		else:
			raise ValueError("Unsupported file format. Use .pt or .h5")


def main():
	parser = argparse.ArgumentParser(description="Sudoku generation CLI")
	parser.add_argument("--amount", type=int, required=True, help="How many puzzles to generate")
	parser.add_argument("--onehot", action="store_true", help="Save in one-hot format (default: False)")
	parser.add_argument("--path", type=str, required=True, help="Save path (should end with .pt or .h5)")

	args = parser.parse_args()

	if not args.path.endswith(".pt") and not args.path.endswith(".h5"):
		print("Error: --path must end with .pt or .h5")
		exit(1)

	gen = SudokuGenerator()
	puzzles, solutions, hole_counts, ids = gen.generate_puzzles(num_puzzles=args.amount)
	gen.save_dataset(puzzles, solutions, hole_counts, ids, path=args.path, one_hot=args.onehot)

	print(f"Generated {args.amount} puzzles and saved to {args.path} (onehot={args.onehot})")


if __name__ == "__main__":
	main()
