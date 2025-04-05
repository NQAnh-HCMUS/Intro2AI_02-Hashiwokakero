from pysat.formula import CNF, IDPool
from pysat.card import CardEnc
#from pysat.solvers import Solver   # Không dùng solver của PySAT nữa
import os
import itertools
import heapq
import time
import psutil 

###########################
# PHẦN SINH CNF (giữ nguyên)
###########################
def read_puzzle(matrix):
    """
    Đọc puzzle dưới dạng ma trận; mỗi ô khác 0 là đảo với số cầu cần nối.
    Trả về dictionary: island_id -> (row, col, required bridges)
    """
    islands = {}
    island_id = 1
    for i, row in enumerate(matrix):
        for j, cell in enumerate(row):
            if cell != 0:
                islands[island_id] = (i, j, cell)
                island_id += 1
    return islands

def compute_edges(islands, matrix):
    """
    Xác định các kết nối hợp lệ giữa các đảo.
    Chỉ xét theo hướng phải (hàng) và xuống (cột) để tránh trùng lặp.
    Trả về:
      - edges: danh sách các tuple (id1, id2, extra) với extra chứa thông tin dạng:
          * ('h', r, c_start, c_end) nếu kết nối ngang
          * ('v', c, r_start, r_end) nếu kết nối dọc
      - coord_to_id: ánh xạ từ tọa độ (row, col) sang id đảo.
    """
    edges = []
    coord_to_id = {}
    for id, (r, c, req) in islands.items():
        coord_to_id[(r, c)] = id
    rows = len(matrix)
    cols = len(matrix[0])
    for id, (r, c, req) in islands.items():
        # Xét theo hàng: tìm đảo liền bên phải
        for nc in range(c+1, cols):
            if (r, nc) in coord_to_id:
                valid = True
                for cc in range(c+1, nc):
                    if matrix[r][cc] != 0:
                        valid = False
                        break
                if valid:
                    id2 = coord_to_id[(r, nc)]
                    edge = tuple(sorted((id, id2)))
                    edges.append( (edge, ('h', r, c, nc)) )
                break  # chỉ xét đảo liền kề đầu tiên
        # Xét theo cột: tìm đảo liền bên dưới
        for nr in range(r+1, rows):
            if (nr, c) in coord_to_id:
                valid = True
                for rr in range(r+1, nr):
                    if matrix[rr][c] != 0:
                        valid = False
                        break
                if valid:
                    id2 = coord_to_id[(nr, c)]
                    edge = tuple(sorted((id, id2)))
                    edges.append( (edge, ('v', c, r, nr)) )
                break
    # Loại bỏ các cạnh trùng lặp
    unique_edges = {}
    for (edge, extra) in edges:
        if edge not in unique_edges:
            unique_edges[edge] = extra
    final_edges = [(edge[0], edge[1], extra) for edge, extra in unique_edges.items()]
    return final_edges, coord_to_id

def add_domain_constraints(cnf, vpool, edges):
    """
    Với mỗi cạnh (edge), tạo ra 2 biến:
      - x_e1: có ít nhất 1 cầu
      - x_e2: cầu thứ hai (chỉ có thể bật khi x_e1 bật)
    Ràng buộc: x_e2 -> x_e1 tương đương (-x_e2 v x_e1).
    Trả về ánh xạ edge_vars: (id1, id2) -> (x_e1, x_e2)
    """
    edge_vars = {}
    for (i, j, extra) in edges:
        x1 = vpool.id(('x', i, j, 1))
        x2 = vpool.id(('x', i, j, 2))
        cnf.append([-x2, x1])
        edge_vars[(i, j)] = (x1, x2)
    return edge_vars

def add_island_constraints(cnf, vpool, islands, edge_vars):
    """
    Với mỗi đảo, tổng số cầu đến đảo phải bằng số yêu cầu.
    Mỗi cạnh nối đến đảo đóng góp: 1 nếu x_e1 bật, cộng thêm 1 nếu x_e2 bật.
    Sử dụng CardEnc.equals để tạo ràng buộc "bằng" một cách tối ưu.
    """
    island_edges = {id: [] for id in islands.keys()}
    for (i, j), (x1, x2) in edge_vars.items():
        if i in island_edges:
            island_edges[i].extend([x1, x2])
        if j in island_edges:
            island_edges[j].extend([x1, x2])
    for id, lits in island_edges.items():
        req = islands[id][2]
        if lits:
            card = CardEnc.equals(lits=lits, bound=req, vpool=vpool, encoding=1)
            cnf.extend(card.clauses)
        else:
            if islands[id][2] > 0:
                cnf.append([])
    return

def add_non_crossing_constraints(cnf, vpool, edges, islands):
    """
    Với mỗi cặp cạnh, nếu một cạnh chạy ngang và một cạnh chạy dọc giao nhau,
    thêm mệnh đề: không cho phép cả 2 đều có cầu (x_e1 của mỗi cạnh không cùng bật).
    """
    edge_list = [((i, j), extra) for (i, j, extra) in edges]
    n = len(edge_list)
    for idx1 in range(n):
        (edge1, extra1) = edge_list[idx1]
        for idx2 in range(idx1+1, n):
            (edge2, extra2) = edge_list[idx2]
            if extra1[0]=='h' and extra2[0]=='v':
                r = extra1[1]
                c_start, c_end = extra1[2], extra1[3]
                c = extra2[1]
                r_start, r_end = extra2[2], extra2[3]
                if r_start < r < r_end and c_start < c < c_end:
                    x1_e1 = vpool.id(('x', edge1[0], edge1[1], 1))
                    x1_e2 = vpool.id(('x', edge2[0], edge2[1], 1))
                    cnf.append([-x1_e1, -x1_e2])
            elif extra1[0]=='v' and extra2[0]=='h':
                r = extra2[1]
                c_start, c_end = extra2[2], extra2[3]
                c = extra1[1]
                r_start, r_end = extra1[2], extra1[3]
                if r_start < r < r_end and c_start < c < c_end:
                    x1_e1 = vpool.id(('x', edge1[0], edge1[1], 1))
                    x1_e2 = vpool.id(('x', edge2[0], edge2[1], 1))
                    cnf.append([-x1_e1, -x1_e2])
    return

def check_connectivity(solution, islands):
    """
    Kiểm tra xem đồ thị các đảo (nodes) với các cầu trong solution có liên thông hay không.
    Dựng đồ thị từ solution (chỉ xét các cạnh có cầu >= 1) và thực hiện DFS.
    """
    graph = {i: [] for i in islands.keys()}
    for (i, j), count in solution.items():
        if count > 0:
            graph[i].append(j)
            graph[j].append(i)
    visited = set()
    def dfs(u):
        visited.add(u)
        for v in graph[u]:
            if v not in visited:
                dfs(v)
    start = min(islands.keys())
    dfs(start)
    return len(visited) == len(islands)

def build_cnf(matrix):
    """Sinh CNF từ puzzle, trả về (cnf, vpool, islands, edges, edge_vars)"""
    islands = read_puzzle(matrix)
    edges, coord_to_id = compute_edges(islands, matrix)
    cnf = CNF()
    vpool = IDPool()
    edge_vars = add_domain_constraints(cnf, vpool, edges)
    add_island_constraints(cnf, vpool, islands, edge_vars)
    add_non_crossing_constraints(cnf, vpool, edges, islands)
    return cnf, vpool, islands, edges, edge_vars

########################################
# PHẦN SOLVER TỰ CÀI (không dùng PySAT)
########################################
# Hỗ trợ: Tính số biến trong CNF
def get_n_vars(cnf):
    n = 0
    for clause in cnf.clauses:
        for lit in clause:
            n = max(n, abs(lit))
    return n

# Hàm chuyển model (assignment) thành solution của Hashiwokakero dựa trên edge_vars
def interpret_model(assignment, edge_vars):
    """
    assignment: list với index 1..n (assignment[i] = True/False)
    Trả về dictionary: (i, j) -> số cầu (1 hoặc 2) cho mỗi edge
    """
    sol = {}
    for key, (x1, x2) in edge_vars.items():
        # Nếu biến x1 được gán True, thì có ít nhất 1 cầu, và nếu x2 True thì 2 cầu
        if assignment[x1]:
            count = 1
            if assignment[x2]:
                count = 2
            sol[key] = count
    return sol

# ---------- Brute-force -----------
def solve_cnf_bruteforce(cnf, edge_vars):
    clauses = cnf.clauses
    n = get_n_vars(cnf)
    # Duyệt qua tất cả 2^n tổ hợp (chỉ dùng cho bài toán nhỏ)
    for combo in itertools.product([False, True], repeat=n):
        # Ta tạo assignment list có độ dài n+1 (index 1..n; bỏ index 0)
        assignment = [None] + list(combo)
        satisfied = True
        for clause in clauses:
            clause_sat = False
            for lit in clause:
                var = abs(lit)
                val = assignment[var]
                if lit > 0 and val is True:
                    clause_sat = True
                    break
                if lit < 0 and val is False:
                    clause_sat = True
                    break
            if not clause_sat:
                satisfied = False
                break
        if satisfied:
            # Chuyển model thành solution
            sol = interpret_model(assignment, edge_vars)
            if check_connectivity(sol, read_puzzle_from_cnf(cnf)):  # dùng hàm trợ giúp để lấy islands
                return assignment  # Trả về assignment dạng list (index 1..n)
    return None

# ---------- Backtracking -----------
def solve_cnf_backtracking(cnf, edge_vars):
    clauses = cnf.clauses
    n = get_n_vars(cnf)
    assignment = [None] * (n+1)  # index 1..n

    def backtrack(i):
        if i > n:
            # Kiểm tra toàn bộ công thức
            for clause in clauses:
                if not any(((lit > 0 and assignment[abs(lit)] is True) or (lit < 0 and assignment[abs(lit)] is False)) for lit in clause):
                    return None
            # Nếu đạt, kiểm tra liên thông
            sol = interpret_model(assignment, edge_vars)
            if check_connectivity(sol, read_puzzle_from_cnf(cnf)):
                return assignment.copy()
            return None
        for val in [False, True]:
            assignment[i] = val
            # Kiểm tra một số mệnh đề có thể xung đột sớm
            conflict = False
            for clause in clauses:
                # Nếu mệnh đề không có literal nào True và không có literal chưa gán thì xung đột
                sat = False
                unassigned = False
                for lit in clause:
                    if assignment[abs(lit)] is None:
                        unassigned = True
                    elif (lit > 0 and assignment[abs(lit)] is True) or (lit < 0 and assignment[abs(lit)] is False):
                        sat = True
                        break
                if not sat and not unassigned:
                    conflict = True
                    break
            if conflict:
                continue
            result = backtrack(i+1)
            if result is not None:
                return result
        assignment[i] = None
        return None

    return backtrack(1)

# ---------- A* -----------
def solve_cnf_astar(cnf, edge_vars):
    clauses = cnf.clauses
    n = get_n_vars(cnf)
    # Mỗi trạng thái: (f, g, assignment), assignment là list độ dài n+1
    init = ([None] * (n+1))
    # Hàm heuristic: đếm số clause chưa thỏa
    def heuristic(assgn):
        h = 0
        for clause in clauses:
            sat = False
            for lit in clause:
                if assgn[abs(lit)] is None:
                    sat = True  # chưa xác định => không tính
                    break
                elif (lit > 0 and assgn[abs(lit)] is True) or (lit < 0 and assgn[abs(lit)] is False):
                    sat = True
                    break
            if not sat:
                h += 1
        return h
    g0 = 0
    h0 = heuristic(init)
    open_heap = []
    heapq.heappush(open_heap, (g0+h0, g0, init))
    closed = set()
    while open_heap:
        f, g, assgn = heapq.heappop(open_heap)
        state_key = tuple(assgn)
        if state_key in closed:
            continue
        closed.add(state_key)
        # Nếu tất cả biến đã được gán, kiểm tra CNF
        if None not in assgn[1:]:
            valid = True
            for clause in clauses:
                if not any((lit > 0 and assgn[abs(lit)] is True) or (lit < 0 and assgn[abs(lit)] is False) for lit in clause):
                    valid = False
                    break
            if valid:
                sol = interpret_model(assgn, edge_vars)
                if check_connectivity(sol, read_puzzle_from_cnf(cnf)):
                    return assgn
            continue
        # Chọn biến đầu tiên chưa gán
        try:
            i = assgn.index(None, 1)
        except ValueError:
            i = None
        if i is None:
            continue
        for val in [False, True]:
            new_assgn = assgn.copy()
            new_assgn[i] = val
            # Kiểm tra sớm: nếu có clause nào hoàn toàn False, bỏ qua
            conflict = False
            for clause in clauses:
                sat = False
                unassigned = False
                for lit in clause:
                    if new_assgn[abs(lit)] is None:
                        unassigned = True
                        continue
                    elif (lit > 0 and new_assgn[abs(lit)] is True) or (lit < 0 and new_assgn[abs(lit)] is False):
                        sat = True
                        break
                if not sat and not unassigned:
                    conflict = True
                    break
            if conflict:
                continue
            new_g = g + 1
            new_h = heuristic(new_assgn)
            heapq.heappush(open_heap, (new_g+new_h, new_g, new_assgn))
    return None

# Hàm trợ giúp: lấy dữ liệu islands từ CNF (giả sử CNF được tạo bởi build_cnf)
def read_puzzle_from_cnf(cnf):
    # Ở đây, ta giả sử rằng CNF được tạo từ puzzle ban đầu và các hàm read_puzzle, …
    # Nếu cần, ta có thể lưu thêm thông tin puzzle trong CNF, nhưng để đơn giản, ta dùng một biến toàn cục.
    # Trong demo này, ta giả sử rằng biến global 'global_islands' đã được thiết lập sau khi build_cnf.
    global global_islands
    return global_islands

########################################
# PHẦN XUẤT KẾT QUẢ (in và ghi file)
########################################
def get_output_string(matrix, islands, edges, solution_dict):
    """
    Trả về chuỗi output với định dạng:
    [ "cell1" , "cell2" , ... ]
    cho mỗi hàng của bản đồ kết quả. Các ô trống hiển thị "0".
    """
    rows = len(matrix)
    cols = len(matrix[0])
    grid = [['0' for _ in range(cols)] for _ in range(rows)]
    
    island_positions = {}
    for id, (r, c, req) in islands.items():
        grid[r][c] = str(req)
        island_positions[id] = (r, c)
    
    for (i, j), count in solution_dict.items():
        r1, c1 = island_positions[i]
        r2, c2 = island_positions[j]
        if r1 == r2:
            for c in range(min(c1, c2)+1, max(c1, c2)):
                grid[r1][c] = '-' if count == 1 else '='
        elif c1 == c2:
            for r in range(min(r1, r2)+1, max(r1, r2)):
                grid[r][c1] = '|' if count == 1 else '$'
    
    lines = []
    for row in grid:
        quoted = [f'"{cell}"' for cell in row]
        lines.append("[ " + " , ".join(quoted) + " ]")
    return "\n".join(lines)

def write_output(output_folder, input_number, output_str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = os.path.join(output_folder, f"output-{input_number}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(output_str)

def print_solution(matrix, islands, edges, solution_dict):
    rows = len(matrix)
    cols = len(matrix[0])
    grid = [['0' for _ in range(cols)] for _ in range(rows)]
    
    island_positions = {}
    for id, (r, c, req) in islands.items():
        grid[r][c] = str(req)
        island_positions[id] = (r, c)
    
    for (i, j), count in solution_dict.items():
        r1, c1 = island_positions[i]
        r2, c2 = island_positions[j]
        if r1 == r2:
            for c in range(min(c1, c2)+1, max(c1, c2)):
                grid[r1][c] = '-' if count == 1 else '='
        elif c1 == c2:
            for r in range(min(r1, r2)+1, max(r1, r2)):
                grid[r][c1] = '|' if count == 1 else '$'
    
    for row in grid:
        print(" ".join(row))

########################################
# PHẦN MAIN: chọn solver và ghi kết quả
########################################
def main():
    global global_islands  # Dùng để lưu lại islands từ build_cnf, phục vụ cho check_connectivity
    
    # Hiển thị menu để người dùng chọn solver
    print("Chọn solver để giải puzzle:")
    print("1: A*")
    print("2: Bruteforce")
    print("3: Backtracking")
    choice = input("Nhập lựa chọn của bạn (1/2/3): ").strip()
    if choice == '1':
        solver_func, solver_name = solve_cnf_astar, "astar"
    elif choice == '2':
        solver_func, solver_name = solve_cnf_bruteforce, "bruteforce"
    elif choice == '3':
        solver_func, solver_name = solve_cnf_backtracking, "backtracking"
    else:
        print("Lựa chọn không hợp lệ. Thoát chương trình.")
        return

    output_folder = f"output_{solver_name}"
    process = psutil.Process(os.getpid())

    # Ví dụ: chạy trên các file input-1.txt và input-3.txt (bạn có thể mở rộng danh sách)
    for idx in [1, 3 , 5]:
        filename = f"input/input-{idx}.txt"
        if not os.path.exists(filename):
            print(f"File {filename} không tồn tại, bỏ qua.")
            continue
        print(f"\n=== Đang giải puzzle {idx} từ file {filename} sử dụng solver {solver_name} ===")
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        matrix = [list(map(int, line.split(","))) for line in lines]
        
        # Sinh CNF và các dữ liệu liên quan
        cnf, vpool, islands, edges, edge_vars = build_cnf(matrix)
        global_islands = islands  # Lưu lại thông tin đảo dùng cho check_connectivity
        
        # Đo thời gian và bộ nhớ trước khi giải
        start_time = time.perf_counter()
        start_mem = process.memory_info().rss
        
        assignment = solver_func(cnf, edge_vars)
        
        # Đo thời gian và bộ nhớ sau khi giải
        end_time = time.perf_counter()
        end_mem = process.memory_info().rss
        
        elapsed_time = end_time - start_time
        memory_used_kb = (end_mem - start_mem) / 1024  # tính theo KB
        
        if assignment is None:
            print("Không tìm được lời giải.")
            output_str = "Không tìm được lời giải cho puzzle."
        else:
            solution_dict = interpret_model(assignment, edge_vars)
            print("Bản đồ kết quả:")
            print_solution(matrix, islands, edges, solution_dict)
            output_str = get_output_string(matrix, islands, edges, solution_dict)
        
        write_output(output_folder, idx, output_str)
        print(f"Kết quả được ghi vào file: {output_folder}/output-{idx}.txt")
        print(f"Thời gian solver {solver_name}: {elapsed_time:.6f} giây")
        print(f"Bộ nhớ sử dụng: {memory_used_kb:.2f} KB")

if __name__ == '__main__':
    main()
# ---------- IMPLEMENTATION CỦA 3 SOLVER TỰ CÀI ----------
def solve_cnf_bruteforce(cnf, edge_vars):
    clauses = cnf.clauses
    n = get_n_vars(cnf)
    for combo in itertools.product([False, True], repeat=n):
        assignment = [None] + list(combo)
        valid = True
        for clause in clauses:
            if not any((lit > 0 and assignment[abs(lit)] is True) or (lit < 0 and assignment[abs(lit)] is False) for lit in clause):
                valid = False
                break
        if valid:
            sol = interpret_model(assignment, edge_vars)
            if check_connectivity(sol, read_puzzle_from_cnf(cnf)):
                return assignment
    return None

def solve_cnf_backtracking(cnf, edge_vars):
    clauses = cnf.clauses
    n = get_n_vars(cnf)
    assignment = [None] * (n+1)
    def backtrack(i):
        if i > n:
            for clause in clauses:
                if not any((lit > 0 and assignment[abs(lit)] is True) or (lit < 0 and assignment[abs(lit)] is False) for lit in clause):
                    return None
            sol = interpret_model(assignment, edge_vars)
            if check_connectivity(sol, read_puzzle_from_cnf(cnf)):
                return assignment.copy()
            return None
        for val in [False, True]:
            assignment[i] = val
            conflict = False
            for clause in clauses:
                sat = False
                unassigned = False
                for lit in clause:
                    if assignment[abs(lit)] is None:
                        unassigned = True
                    elif (lit > 0 and assignment[abs(lit)] is True) or (lit < 0 and assignment[abs(lit)] is False):
                        sat = True
                        break
                if not sat and not unassigned:
                    conflict = True
                    break
            if conflict:
                continue
            result = backtrack(i+1)
            if result is not None:
                return result
        assignment[i] = None
        return None
    return backtrack(1)

def solve_cnf_astar(cnf, edge_vars):
    clauses = cnf.clauses
    n = get_n_vars(cnf)
    init = [None] * (n+1)
    def heuristic(assgn):
        h = 0
        for clause in clauses:
            if not any((lit > 0 and assgn[abs(lit)] is True) or (lit < 0 and assgn[abs(lit)] is False) for lit in clause):
                h += 1
        return h
    g0 = 0
    h0 = heuristic(init)
    open_heap = []
    heapq.heappush(open_heap, (g0+h0, g0, init))
    closed = set()
    while open_heap:
        f, g, assgn = heapq.heappop(open_heap)
        state_key = tuple(assgn)
        if state_key in closed:
            continue
        closed.add(state_key)
        if None not in assgn[1:]:
            if all(any((lit > 0 and assgn[abs(lit)] is True) or (lit < 0 and assgn[abs(lit)] is False) for lit in clause) for clause in clauses):
                sol = interpret_model(assgn, edge_vars)
                if check_connectivity(sol, read_puzzle_from_cnf(cnf)):
                    return assgn
            continue
        try:
            i = assgn.index(None, 1)
        except ValueError:
            continue
        for val in [False, True]:
            new_assgn = assgn.copy()
            new_assgn[i] = val
            conflict = False
            for clause in clauses:
                sat = False
                unassigned = False
                for lit in clause:
                    if new_assgn[abs(lit)] is None:
                        unassigned = True
                        continue
                    elif (lit > 0 and new_assgn[abs(lit)] is True) or (lit < 0 and new_assgn[abs(lit)] is False):
                        sat = True
                        break
                if not sat and not unassigned:
                    conflict = True
                    break
            if conflict:
                continue
            new_g = g + 1
            new_h = heuristic(new_assgn)
            heapq.heappush(open_heap, (new_g+new_h, new_g, new_assgn))
    return None

def read_puzzle_from_cnf(cnf):
    # Ta lưu lại thông tin islands trong biến toàn cục
    global global_islands
    return global_islands

# Nếu chạy file này trực tiếp:
if __name__ == '__main__':
    main()
