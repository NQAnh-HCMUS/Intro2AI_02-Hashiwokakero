from Hashiwokakero import *

# Example usage
if __name__ == "__main__":
    # Example puzzle input
    puzzle_input = """
    0, 2, 0, 5, 0, 0, 2
    0, 0, 0, 0, 0, 0, 0
    4, 0, 2, 0, 2, 0, 4
    0, 0, 0, 0, 0, 0, 0
    0, 1, 0, 5, 0, 2, 0
    0, 0, 0, 0, 0, 0, 0
    4, 0, 0, 0, 0, 0, 3
    """
    # puzzle = [
    #     [0, 2, 0, 5, 0, 0, 2],
    #     [0, 0, 0, 0, 0, 0, 0], 
    #     [4, 0, 2, 0, 2, 0, 4],
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 5, 0, 2, 0],
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [4, 0, 0, 0, 0, 0, 3]
    # ]
    puzzle = [
        [2,0,1,0,0],
        [0,0,0,0,0],
        [4,0,3,0,1],
        [0,0,0,0,0],
        [3,0,0,0,2]
    ]
    
    try:
        puzzle = read_puzzle_from_input(puzzle_input)
    except ValueError as e:
        print(f"Error: {e}")
        exit()
    
    solver = HashiwokakeroSolver(puzzle)
    
    print("Solving puzzle:")
    for row in puzzle:
        print(' '.join(str(x) if x > 0 else '.' for x in row))
    print()
    
    if solver.solve():
        print("Solution found:")
        solution = solver.print_solution()
        for row in solution:
            print(f" [\"{'\", \"'.join(row)}\"]")
    else:
        print("No solution exists for this puzzle.")