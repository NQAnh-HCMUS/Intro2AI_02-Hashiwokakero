import numpy as np
from typing import List, Tuple, Optional

class HashiwokakeroSolver:
    def __init__(self, grid: List[List[int]]):
        """Initialize the solver with the puzzle grid.
        
        Args:
            grid: 2D list representing the puzzle. 0 means no island, positive numbers
                  indicate islands with their connection requirements.
        """
        self.original_grid = np.array(grid, dtype=int)
        self.grid = self.original_grid.copy()
        self.rows, self.cols = self.grid.shape
        self.bridges = np.zeros((self.rows, self.cols, 2), dtype=int)  # 0: horizontal, 1: vertical
        self.islands = self._find_islands()
        
    def _find_islands(self) -> List[Tuple[int, int, int]]:
        """Find all islands in the grid and return their coordinates and required connections."""
        islands = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] > 0:
                    islands.append((i, j, self.grid[i, j]))
        return islands
    
    def _get_possible_bridges(self, i: int, j: int) -> List[Tuple[int, int, int, int]]:
        """Find possible bridges from island at (i,j).
        
        Returns:
            List of (end_i, end_j, direction, max_bridges) tuples
        """
        possible = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            while 0 <= ni < self.rows and 0 <= nj < self.cols:
                if self.grid[ni, nj] > 0:  # found another island
                    # Check if bridge would be horizontal (0) or vertical (1)
                    direction = 0 if di == 0 else 1
                    max_possible = min(self.grid[i, j], self.grid[ni, nj])
                    possible.append((ni, nj, direction, max_possible))
                    break
                ni += di
                nj += dj
                
        return possible
    
    def _is_bridge_possible(self, i1: int, j1: int, i2: int, j2: int, direction: int) -> bool:
        """Check if a bridge between two islands is possible without crossing existing bridges."""
        if direction == 0:  # horizontal bridge
            if i1 != i2:
                return False
            step = 1 if j2 > j1 else -1
            for j in range(j1 + step, j2, step):
                # Check for crossing vertical bridges
                if self.bridges[i1, j, 1] > 0:
                    return False
                # Check for existing horizontal bridges (would be parallel)
                if self.bridges[i1, j, 0] > 0 and (i1, j) != (i1, j1) and (i1, j) != (i2, j2):
                    return False
        else:  # vertical bridge
            if j1 != j2:
                return False
            step = 1 if i2 > i1 else -1
            for i in range(i1 + step, i2, step):
                # Check for crossing horizontal bridges
                if self.bridges[i, j1, 0] > 0:
                    return False
                # Check for existing vertical bridges (would be parallel)
                if self.bridges[i, j1, 1] > 0 and (i, j1) != (i1, j1) and (i, j1) != (i2, j2):
                    return False
        return True
    
    def _add_bridge(self, i1: int, j1: int, i2: int, j2: int, direction: int, count: int = 1) -> bool:
        """Add a bridge between two islands."""
        if not self._is_bridge_possible(i1, j1, i2, j2, direction):
            return False
        
        if direction == 0:  # horizontal
            step = 1 if j2 > j1 else -1
            for j in range(j1, j2 + step, step):
                self.bridges[i1, j, 0] += count
        else:  # vertical
            step = 1 if i2 > i1 else -1
            for i in range(i1, i2 + step, step):
                self.bridges[i, j1, 1] += count
        
        # Update island connection counts
        self.grid[i1, j1] -= count
        self.grid[i2, j2] -= count
        return True
    
    def _remove_bridge(self, i1: int, j1: int, i2: int, j2: int, direction: int, count: int = 1) -> None:
        """Remove a bridge between two islands."""
        if direction == 0:  # horizontal
            step = 1 if j2 > j1 else -1
            for j in range(j1, j2 + step, step):
                self.bridges[i1, j, 0] -= count
        else:  # vertical
            step = 1 if i2 > i1 else -1
            for i in range(i1, i2 + step, step):
                self.bridges[i, j1, 1] -= count
        
        # Restore island connection counts
        self.grid[i1, j1] += count
        self.grid[i2, j2] += count
    
    def _is_solved(self) -> bool:
        """Check if the puzzle is solved (all islands have 0 required connections)."""
        return np.all(self.grid == 0) and self._is_fully_connected()
    
    def _is_fully_connected(self) -> bool:
        """Check if all islands are connected as a single group."""
        if not self.islands:
            return True
            
        visited = set()
        stack = [self.islands[0][:2]]  # Start with first island
        
        while stack:
            i, j = stack.pop()
            if (i, j) in visited:
                continue
            visited.add((i, j))
            
            # Check all four directions for bridges
            # Right
            if j + 1 < self.cols and self.bridges[i, j, 0] > 0:
                ni, nj = i, j + 1
                while nj < self.cols and self.bridges[i, nj, 0] == 0:
                    nj += 1
                if nj < self.cols and self.grid[i, nj] > 0:
                    stack.append((i, nj))
            
            # Left
            if j - 1 >= 0 and self.bridges[i, j - 1, 0] > 0:
                ni, nj = i, j - 1
                while nj >= 0 and self.bridges[i, nj, 0] == 0:
                    nj -= 1
                if nj >= 0 and self.grid[i, nj] > 0:
                    stack.append((i, nj))
            
            # Down
            if i + 1 < self.rows and self.bridges[i, j, 1] > 0:
                ni, nj = i + 1, j
                while ni < self.rows and self.bridges[ni, j, 1] == 0:
                    ni += 1
                if ni < self.rows and self.grid[ni, j] > 0:
                    stack.append((ni, j))
            
            # Up
            if i - 1 >= 0 and self.bridges[i - 1, j, 1] > 0:
                ni, nj = i - 1, j
                while ni >= 0 and self.bridges[ni, j, 1] == 0:
                    ni -= 1
                if ni >= 0 and self.grid[ni, j] > 0:
                    stack.append((ni, j))
        
        # Check if all islands were visited
        return len(visited) == len([isl for isl in self.islands if self.grid[isl[0], isl[1]] > 0])
    
    def solve(self) -> bool:
        """Solve the puzzle using backtracking."""
        if self._is_solved():
            return True
            
        # Find the island with the fewest remaining connections but at least 1
        min_connections = float('inf')
        selected_island = None
        possible_bridges = None
        
        for island in self.islands:
            i, j, req = island
            if 0 < self.grid[i, j] < min_connections:
                possible = self._get_possible_bridges(i, j)
                if possible:  # Only consider if there are possible bridges
                    min_connections = self.grid[i, j]
                    selected_island = (i, j)
                    possible_bridges = possible
        
        if not selected_island:
            return False
            
        i, j = selected_island
        
        for ni, nj, direction, max_bridges in possible_bridges:
            for count in [min(2, max_bridges), 1]:  # Try double bridge first if possible
                if count > self.grid[i, j] or count > self.grid[ni, nj]:
                    continue
                    
                if self._add_bridge(i, j, ni, nj, direction, count):
                    if self.solve():
                        return True
                    self._remove_bridge(i, j, ni, nj, direction, count)
        
        return False
    
    def print_solution(self) -> None:
        """Print the solved puzzle with bridges."""
        if not self._is_solved():
            print("Puzzle not solved yet!")
            return
            
        # Create a display grid
        display = [[' ' for _ in range(2 * self.cols - 1)] for _ in range(2 * self.rows - 1)]
        
        # Mark islands
        for i in range(self.rows):
            for j in range(self.cols):
                if self.original_grid[i, j] > 0:
                    display[2 * i][2 * j] = str(self.original_grid[i, j])
        
        # Draw horizontal bridges
        for i in range(self.rows):
            for j in range(self.cols - 1):
                if self.bridges[i, j, 0] > 0:
                    display[2 * i][2 * j + 1] = '=' if self.bridges[i, j, 0] == 2 else '-'
        
        # Draw vertical bridges
        for i in range(self.rows - 1):
            for j in range(self.cols):
                if self.bridges[i, j, 1] > 0:
                    display[2 * i + 1][2 * j] = '‖' if self.bridges[i, j, 1] == 2 else '|'
        
        # Print the display
        for row in display:
            print(' '.join(row))
        print()


def read_puzzle_from_file(filename: str) -> List[List[int]]:
    """Read a Hashiwokakero puzzle from a text file.
    
    Args:
        filename: Path to the text file containing the puzzle
        
    Returns:
        2D list representing the puzzle grid
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    grid = []
    for line in lines:
        row = []
        for char in line:
            if char.isdigit():
                row.append(int(char))
            else:
                row.append(0)
        grid.append(row)
    
    # Validate that all rows have the same length
    row_lengths = [len(row) for row in grid]
    if len(set(row_lengths)) > 1:
        raise ValueError("All rows in the puzzle must have the same length")
    
    return grid


