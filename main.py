from Hashiwokakero import *

# Example usage
if __name__ == "__main__":
    # Read puzzle from file
    try:
        puzzle = read_puzzle_from_file("input-01.txt")
    except FileNotFoundError:
        print("Error: Puzzle file not found. Using default puzzle instead.")
        # Default puzzle if file not found
        puzzle = [
            [2, 0, 4, 0, 3],
            [0, 0, 0, 0, 0],
            [3, 0, 5, 0, 2],
            [0, 0, 0, 0, 0],
            [1, 0, 3, 0, 2]
        ]
    
    solver = HashiwokakeroSolver(puzzle)
    print("Solving puzzle:")
    for row in puzzle:
        print(' '.join(str(x) if x > 0 else '.' for x in row))
    print()
    
    if solver.solve():
        print("Solution found:")
        solver.print_solution()
    else:
        print("No solution exists for this puzzle.")
