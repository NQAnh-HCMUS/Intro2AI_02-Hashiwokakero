import math
import heapq
import time
# 1. Tiền xử lý: đọc puzzle, tính các kết nối hợp lệ, ánh xạ đảo đến edge, và tính cặp giao nhau

def read_puzzle(matrix):
    """
    Đọc puzzle từ ma trận; các ô khác 0 là đảo với số cầu cần nối.
    Trả về: dictionary island_id -> (row, col, required bridges)
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
    Chỉ xét theo hướng phải và xuống để tránh trùng lặp.
    Trả về: 
       - edges: danh sách các tuple (id1, id2, extra)
         extra = ('h', r, c_start, c_end) hoặc ('v', c, r_start, r_end)
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
                    edges.append( (edge[0], edge[1], ('h', r, c, nc)) )
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
                    edges.append( (edge[0], edge[1], ('v', c, r, nr)) )
                break
    return edges, coord_to_id

def build_island_edge_map(edges, islands):
    """
    Xây dựng ánh xạ từ mỗi đảo sang danh sách chỉ số (index) của edge liên quan.
    """
    island_to_edges = {island_id: [] for island_id in islands.keys()}
    for idx, (i, j, extra) in enumerate(edges):
        island_to_edges[i].append(idx)
        island_to_edges[j].append(idx)
    return island_to_edges

def compute_crossing_pairs(edges):
    """
    Tính danh sách các cặp edge giao nhau (dựa vào thông tin extra).
    Nếu một edge ngang và một edge dọc giao nhau thì không được phép cùng có cầu (>0).
    Trả về danh sách các tuple (index_edge1, index_edge2).
    """
    crossing = []
    edge_extras = [extra for (_, _, extra) in edges]
    for idx1, extra1 in enumerate(edge_extras):
        for idx2 in range(idx1+1, len(edge_extras)):
            extra2 = edge_extras[idx2]
            if extra1[0]=='h' and extra2[0]=='v':
                r = extra1[1]
                c_start, c_end = extra1[2], extra1[3]
                c = extra2[1]
                r_start, r_end = extra2[2], extra2[3]
                if r_start < r < r_end and c_start < c < c_end:
                    crossing.append((idx1, idx2))
            elif extra1[0]=='v' and extra2[0]=='h':
                r = extra2[1]
                c_start, c_end = extra2[2], extra2[3]
                c = extra1[1]
                r_start, r_end = extra1[2], extra1[3]
                if r_start < r < r_end and c_start < c < c_end:
                    crossing.append((idx1, idx2))
    return crossing

# 2. Các hàm kiểm tra tính hợp lệ của trạng thái

def is_valid_state(state, island_to_edges, islands, crossing_pairs):
    """
    Kiểm tra trạng thái partial (state: list với giá trị 0,1,2 hoặc None cho mỗi edge)
    - Với mỗi đảo: tổng số cầu hiện tại không vượt quá yêu cầu và có khả năng đạt đủ (dựa vào số edge chưa gán)
    - Kiểm tra không có 2 edge giao nhau mà cùng có cầu (>0)
    """
    # Kiểm tra ràng buộc của mỗi đảo
    for island_id, (r, c, req) in islands.items():
        indices = island_to_edges[island_id]
        assigned_sum = 0
        unassigned = 0
        for idx in indices:
            if state[idx] is None:
                unassigned += 1
            else:
                assigned_sum += state[idx]
        if assigned_sum > req:
            return False
        if assigned_sum + 2 * unassigned < req:
            return False
    # Kiểm tra ràng buộc không giao nhau:
    for (i, j) in crossing_pairs:
        if state[i] is not None and state[i] > 0 and state[j] is not None and state[j] > 0:
            return False
    return True

def state_to_solution(state, edges):
    """
    Chuyển trạng thái (state) thành dictionary: key là tuple (i,j) của đảo, value là số cầu đã nối.
    Chỉ lấy các edge có giá trị >0.
    """
    solution = {}
    for idx, val in enumerate(state):
        if val is not None and val > 0:
            i, j, extra = edges[idx]
            solution[(i, j)] = val
    return solution

def check_connectivity(solution, islands):
    """
    Kiểm tra tính liên thông của đồ thị đảo dựa vào các edge có cầu.
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
    start = next(iter(islands.keys()))
    dfs(start)
    return len(visited) == len(islands)

# 3. Hàm heuristic cho A*
def heuristic(state, island_to_edges, islands):
    """
    Ước lượng số cầu còn thiếu.
    Tính tổng (required - current_sum) cho tất cả các đảo, chia cho 2 (vì mỗi cầu bổ sung cho 2 đảo).
    """
    total_remaining = 0
    for island_id, (r, c, req) in islands.items():
        s = 0
        for idx in island_to_edges[island_id]:
            if state[idx] is not None:
                s += state[idx]
        total_remaining += max(0, req - s)
    return math.ceil(total_remaining / 2)

# 4. Hàm chọn biến (edge) chưa gán cho thuật toán backtracking
def choose_next_edge(state, island_to_edges, islands, edges):
    """
    Chọn edge chưa gán mà "bị ràng buộc" nhiều nhất.
    Ví dụ: tính slack (số cầu còn thiếu) của hai đảo nối với edge đó,
    chọn edge có giá trị min(slack_i, slack_j) nhỏ nhất.
    """
    best_idx = None
    best_score = float('inf')
    for idx in range(len(state)):
        if state[idx] is None:
            i, j, extra = edges[idx]
            slack_i = islands[i][2] - sum(state[k] for k in island_to_edges[i] if state[k] is not None)
            slack_j = islands[j][2] - sum(state[k] for k in island_to_edges[j] if state[k] is not None)
            score = min(slack_i, slack_j)
            if score < best_score:
                best_score = score
                best_idx = idx
    return best_idx

# 5. Triển khai thuật toán A*

def solve_astar(matrix):
    islands = read_puzzle(matrix)
    edges, coord_to_id = compute_edges(islands, matrix)
    island_to_edges = build_island_edge_map(edges, islands)
    crossing_pairs = compute_crossing_pairs(edges)
    n = len(edges)
    
    # state: danh sách độ dài n, mỗi phần tử là None, 0, 1 hoặc 2.
    initial_state = [None] * n
    # Mỗi phần tử trong open_set: (f, idx, state, g)
    open_set = []
    h_val = heuristic(initial_state, island_to_edges, islands)
    heapq.heappush(open_set, (0 + h_val, 0, initial_state, 0))
    visited = set()
    
    while open_set:
        f, idx, state, g = heapq.heappop(open_set)
        # Nếu đã gán hết các edge, kiểm tra ràng buộc đầy đủ
        if idx == n:
            valid = True
            for island_id, (r, c, req) in islands.items():
                s = sum(state[i] for i in island_to_edges[island_id] if state[i] is not None)
                if s != req:
                    valid = False
                    break
            if not valid:
                continue
            sol = state_to_solution(state, edges)
            if check_connectivity(sol, islands):
                return sol, islands, edges
            else:
                continue
        state_tuple = tuple(x if x is not None else -1 for x in state)
        if (idx, state_tuple) in visited:
            continue
        visited.add((idx, state_tuple))
        # Gán giá trị cho edge tại vị trí idx
        for val in [0, 1, 2]:
            new_state = state.copy()
            new_state[idx] = val
            new_g = g + val
            if not is_valid_state(new_state, island_to_edges, islands, crossing_pairs):
                continue
            new_idx = idx + 1
            h_val = heuristic(new_state, island_to_edges, islands)
            heapq.heappush(open_set, (new_g + h_val, new_idx, new_state, new_g))
    return None, islands, edges

# 6. Triển khai thuật toán vét cạn (brute-force) với DFS đơn giản

def solve_bruteforce(matrix):
    islands = read_puzzle(matrix)
    edges, coord_to_id = compute_edges(islands, matrix)
    island_to_edges = build_island_edge_map(edges, islands)
    crossing_pairs = compute_crossing_pairs(edges)
    n = len(edges)
    solution_found = None

    def recurse(idx, state):
        nonlocal solution_found
        if solution_found is not None:
            return
        if idx == n:
            valid = True
            for island_id, (r, c, req) in islands.items():
                s = sum(state[i] for i in island_to_edges[island_id])
                if s != req:
                    valid = False
                    break
            if not valid:
                return
            sol = state_to_solution(state, edges)
            if check_connectivity(sol, islands):
                solution_found = state.copy()
            return
        for val in [0, 1, 2]:
            state[idx] = val
            if is_valid_state(state, island_to_edges, islands, crossing_pairs):
                recurse(idx+1, state)
            if solution_found is not None:
                return
        state[idx] = None

    state = [None] * n
    recurse(0, state)
    if solution_found is not None:
        sol = state_to_solution(solution_found, edges)
        return sol, islands, edges
    else:
        return None, islands, edges

# 7. Triển khai thuật toán quay lui (backtracking) có chọn biến theo heuristic

def solve_backtracking(matrix):
    islands = read_puzzle(matrix)
    edges, coord_to_id = compute_edges(islands, matrix)
    island_to_edges = build_island_edge_map(edges, islands)
    crossing_pairs = compute_crossing_pairs(edges)
    n = len(edges)
    solution_found = None

    def backtrack(state):
        nonlocal solution_found
        if solution_found is not None:
            return
        if all(x is not None for x in state):
            valid = True
            for island_id, (r, c, req) in islands.items():
                s = sum(state[idx] for idx in island_to_edges[island_id])
                if s != req:
                    valid = False
                    break
            if not valid:
                return
            sol = state_to_solution(state, edges)
            if check_connectivity(sol, islands):
                solution_found = state.copy()
            return
        idx = choose_next_edge(state, island_to_edges, islands, edges)
        if idx is None:
            return
        for val in [0, 1, 2]:
            state[idx] = val
            if is_valid_state(state, island_to_edges, islands, crossing_pairs):
                backtrack(state)
            if solution_found is not None:
                return
            state[idx] = None

    state = [None] * n
    backtrack(state)
    if solution_found is not None:
        sol = state_to_solution(solution_found, edges)
        return sol, islands, edges
    else:
        return None, islands, edges

# 8. Hàm in kết quả (in ra “bản đồ” theo định dạng yêu cầu)
def print_solution(matrix, islands, edges, solution):
    rows = len(matrix)
    cols = len(matrix[0])
    output = [[' ' for _ in range(cols)] for _ in range(rows)]
    
    # Đặt đảo vào vị trí
    island_positions = {}
    for id, (r, c, req) in islands.items():
        output[r][c] = str(req)
        island_positions[id] = (r, c)
    
    # Vẽ cầu
    for (i, j), count in solution.items():
        r1, c1 = island_positions[i]
        r2, c2 = island_positions[j]
        if r1 == r2:
            r = r1
            for c in range(min(c1, c2)+1, max(c1, c2)):
                output[r][c] = '-' if count == 1 else '='
        elif c1 == c2:
            c = c1
            for r in range(min(r1, r2)+1, max(r1, r2)):
                output[r][c] = '|' if count == 1 else '$'
    
    for row in output:
        print(' '.join(row))



def main():
    """
    Đọc 10 puzzle từ file input-1.txt đến input-10.txt (mỗi file là một ma trận),
    giải và in kết quả.
    """
    import os
    
    for idx in range(1, 15):
        filename = f"input/input-{idx}.txt"
        if not os.path.exists(filename):
            print(f"File {filename} không tồn tại, bỏ qua.")
            continue
        
        print(f"\n=== Đang giải puzzle {idx} từ file {filename} ===")
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]  # bỏ dòng trống
        # Mỗi dòng là các số nguyên cách nhau
        matrix = [list(map(int, line.split(","))) for line in lines]
        # A*:
        start_time = time.perf_counter()
        sol_astar, islands, edges = solve_astar(matrix)
        end_time = time.perf_counter()
        print("== Thuật toán A* ==")
        if sol_astar is not None:
            print_solution(matrix, islands, edges, sol_astar)
        else:
            print("Không tìm được lời giải với A*")
        print(f"Thời gian A*: {end_time - start_time:.6f} giây\n")

        print("---------------------------------\n")
        # Chạy thuật toán vét cạn (brute-force)
        sol_brute, islands, edges = solve_bruteforce(matrix)
        start_time = time.perf_counter()
        sol_astar, islands, edges = solve_bruteforce(matrix)
        end_time = time.perf_counter()
        if sol_brute is not None:
            print("\nBản đồ kết quả Brute-force:")
            print_solution(matrix, islands, edges, sol_brute)
        else:
            print("Không tìm được lời giải với brute-force")
        print(f"Thời gian Brute-force: {end_time - start_time:.6f} giây\n")

        print("---------------------------------\n")
        # Chạy thuật toán quay lui (backtracking)
        start_time = time.perf_counter()
        sol_bt, islands, edges = solve_backtracking(matrix)
        end_time = time.perf_counter()
        if sol_bt is not None:
            print("\nBản đồ kết quả Backtracking:")
            print_solution(matrix, islands, edges, sol_bt)
        else:
            print("Không tìm được lời giải với backtracking")
        print(f"Thời gian Backtracking: {end_time - start_time:.6f} giây\n")

# Nếu muốn chạy file này trực tiếp:
if __name__ == '__main__':
    main()
