from pysat.formula import CNF, IDPool
from pysat.card import CardEnc
from pysat.solvers import Solver
import os
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
        # Ràng buộc: nếu x2 được bật thì x1 phải được bật
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
            # Mã hoá ràng buộc "equals req" cho tổng các biến
            card = CardEnc.equals(lits=lits, bound=req, vpool=vpool, encoding=1)
            cnf.extend(card.clauses)
        else:
            # Nếu đảo có yêu cầu > 0 nhưng không có cạnh nối, puzzle không giải được
            if islands[id][2] > 0:
                cnf.append([])  # Mệnh đề rỗng => unsat
    return

def add_non_crossing_constraints(cnf, vpool, edges, islands):
    """
    Với mỗi cặp cạnh, nếu một cạnh chạy ngang và một cạnh chạy dọc giao nhau,
    thêm mệnh đề: không cho phép cả 2 đều có cầu (x_e1 của mỗi cạnh không cùng bật).
    """
    # Mỗi edge: extra = ('h', r, c_start, c_end) hoặc ('v', c, r_start, r_end)
    edge_list = [((i,j), extra) for (i,j,extra) in edges]
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
                # Kiểm tra giao nhau
                if r_start < r < r_end and c_start < c < c_end:
                    x1_e1 = vpool.id(('x', edge1[0], edge1[1], 1))
                    x1_e2 = vpool.id(('x', edge2[0], edge2[1], 1))
                    # Không được bật cùng lúc cả hai cạnh
                    cnf.append([-x1_e1, -x1_e2])
            elif extra1[0]=='v' and extra2[0]=='h':
                r = extra2[1]
                c_start, c_end = extra2[2], extra2[3]
                c = extra1[1]
                r_start, r_end = extra1[2], extra1[3]
                # Kiểm tra giao nhau
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
    # Xây dựng đồ thị: node -> danh sách kề
    graph = {i: [] for i in islands.keys()}
    for (i, j), count in solution.items():
        if count > 0:
            graph[i].append(j)
            graph[j].append(i)
    # DFS để đếm số đảo visited
    visited = set()
    def dfs(u):
        visited.add(u)
        for v in graph[u]:
            if v not in visited:
                dfs(v)
    start = min(islands.keys())
    dfs(start)
    return len(visited) == len(islands)

def solve_hashiwokakero_lazy(matrix):
    """
    Xây dựng CNF cho puzzle (ma trận) và giải bằng PySAT mà KHÔNG mã hoá trực tiếp liên thông.
    Sau đó, kiểm tra liên thông của nghiệm.
    Nếu nghiệm không liên thông, thêm mệnh đề cấm nghiệm đó và tiếp tục tìm nghiệm khác.
    Trả về solution (dictionary edge -> số cầu), islands, edges.
    """
    islands = read_puzzle(matrix)
    edges, coord_to_id = compute_edges(islands, matrix)

    cnf = CNF()
    vpool = IDPool()

    # Bước 1: Tạo biến và ràng buộc miền cho các cạnh
    edge_vars = add_domain_constraints(cnf, vpool, edges)
    # Bước 2: Ràng buộc số cầu của mỗi đảo
    add_island_constraints(cnf, vpool, islands, edge_vars)
    # Bước 3: Ràng buộc không giao nhau
    add_non_crossing_constraints(cnf, vpool, edges, islands)

    solver = Solver(name='glucose3')
    solver.append_formula(cnf)

    while solver.solve():
        model = solver.get_model()
        solution = {}
        used_literals = []  # lưu các biến x_e1 đang bật trong nghiệm hiện tại

        for (i, j), (x1, x2) in edge_vars.items():
            # model[x1-1] > 0 nghĩa là x1 đang "True" (vì chỉ số trong model là x1-1)
            if model[x1-1] > 0:
                count = 1
                if model[x2-1] > 0:  # kiểm tra x2
                    count = 2
                solution[(i, j)] = count
                used_literals.append(x1)

        # Kiểm tra liên thông
        if check_connectivity(solution, islands):
            solver.delete()
            return solution, islands, edges
        else:
            # Nghiệm không liên thông => chặn nghiệm này
            # Thêm mệnh đề cấm tất cả x_e1 đã bật đồng thời
            solver.add_clause([-lit for lit in used_literals])

    solver.delete()
    return None, None, None

def print_solution(matrix, islands, edges, solution):
    """
    In kết quả dưới dạng “bản đồ”:
      - Đảo in số cầu yêu cầu.
      - Cầu nối ngang: '-' (1 cầu), '=' (2 cầu).
      - Cầu nối dọc: '|' (1 cầu), '$' (2 cầu).
    """
    rows = len(matrix)
    cols = len(matrix[0])
    output = [[' ' for _ in range(cols)] for _ in range(rows)]

    # Đặt đảo
    island_positions = {}
    for id, (r, c, req) in islands.items():
        output[r][c] = str(req)
        island_positions[id] = (r, c)

    # Vẽ cầu
    for (i, j), count in solution.items():
        r1, c1 = island_positions[i]
        r2, c2 = island_positions[j]
        if r1 == r2:
            # Cầu ngang
            r = r1
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                output[r][c] = '-' if count == 1 else '='
        elif c1 == c2:
            # Cầu dọc
            c = c1
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                output[r][c] = '|' if count == 1 else '$'

    for row in output:
        print(' '.join(row))



def get_output_string(matrix, islands, edges, solution):
    """
    Trả về chuỗi output với định dạng:
    [ "cell1" , "cell2" , ... ]
    cho mỗi hàng của bản đồ kết quả.
    """
    rows = len(matrix)
    cols = len(matrix[0])
    grid = [[' ' for _ in range(cols)] for _ in range(rows)]
    
    # Đặt đảo vào vị trí
    island_positions = {}
    for id, (r, c, req) in islands.items():
        grid[r][c] = str(req)
        island_positions[id] = (r, c)
    
    # Vẽ cầu
    for (i, j), count in solution.items():
        r1, c1 = island_positions[i]
        r2, c2 = island_positions[j]
        if r1 == r2:
            r = r1
            for c in range(min(c1, c2)+1, max(c1, c2)):
                grid[r][c] = '-' if count == 1 else '='
        elif c1 == c2:
            c = c1
            for r in range(min(r1, r2)+1, max(r1, r2)):
                grid[r][c] = '|' if count == 1 else '$'
    
    # Tạo chuỗi output
    lines = []
    for row in grid:
        # Mỗi phần tử được đưa vào trong dấu ngoặc kép và phân cách bằng " , "
        quoted = [f'"{cell}"' for cell in row]
        line = "[ " + " , ".join(quoted) + " ]"
        lines.append(line)
    return "\n".join(lines)


# 9. Hàm ghi output ra file
def write_output(output_folder, input_number, output_str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = os.path.join(output_folder, f"output-{input_number}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(output_str)


# ------------------------ HÀM MAIN SỬA ĐỔI ------------------------
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
        
        solution, islands, edges = solve_hashiwokakero_lazy(matrix)
        
        if solution is None:
            print("Không tìm được lời giải cho puzzle.")
        else:
            print("\nBản đồ kết quả:")
            print_solution(matrix, islands, edges, solution)

# Nếu muốn chạy file này trực tiếp:
if __name__ == '__main__':
    main()
