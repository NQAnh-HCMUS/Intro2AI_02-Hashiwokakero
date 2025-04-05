from pysat.formula import CNF, IDPool
from pysat.card import CardEnc
from pysat.solvers import Solver

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
            # Dùng encoding=1 (sequential counter) để mã hóa ràng buộc bằng.
            card = CardEnc.equals(lits=lits, bound=req, vpool=vpool, encoding=1)
            cnf.extend(card.clauses)
        else:
            # Nếu đảo có yêu cầu > 0 nhưng không có cạnh nối, puzzle không giải được.
            cnf.append([])  # mệnh đề rỗng => unsat
    return

def add_non_crossing_constraints(cnf, vpool, edges, islands):
    """
    Với mỗi cặp cạnh, nếu một cạnh chạy ngang và một chạy dọc giao nhau,
    thêm mệnh đề: không cho phép cả 2 đều có cầu (x_e1 của mỗi cạnh không cùng bật).
    """
    # Mỗi edge có extra info: ('h', r, c_start, c_end) hoặc ('v', c, r_start, r_end)
    for idx1, (edge1, extra1) in enumerate( [((i,j), extra) for (i,j,extra) in edges] ):
        for idx2, (edge2, extra2) in enumerate( [((i,j), extra) for (i,j,extra) in edges] ):
            if idx2 <= idx1:
                continue
            # Xét trường hợp một cạnh ngang và một cạnh dọc
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
    # Xây dựng đồ thị dưới dạng dictionary: node -> danh sách các node kề.
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

def solve_hashiwokakero_lazy(matrix):
    """
    Xây dựng CNF dựa trên puzzle (ma trận) và giải bằng PySAT mà KHÔNG mã hóa trực tiếp liên thông.
    Sau đó, kiểm tra liên thông của nghiệm.
    Nếu không liên thông, chặn nghiệm đó và tiếp tục giải.
    Trả về solution (dictionary edge -> số cầu), islands, edges.
    """
    islands = read_puzzle(matrix)
    edges, coord_to_id = compute_edges(islands, matrix)
    
    cnf = CNF()
    vpool = IDPool()
    
    # Bước 1: Tạo biến và ràng buộc miền cho các cạnh.
    edge_vars = add_domain_constraints(cnf, vpool, edges)
    
    # Bước 2: Ràng buộc số cầu của mỗi đảo.
    add_island_constraints(cnf, vpool, islands, edge_vars)
    
    # Bước 3: Ràng buộc không giao nhau.
    add_non_crossing_constraints(cnf, vpool, edges, islands)
    
    solver = Solver(name='glucose3')
    solver.append_formula(cnf)
    
    while solver.solve():
        model = solver.get_model()
        solution = {}
        used_literals = []  # Lưu lại các biến x_e1 mà ở nghiệm hiện tại được bật.
        for (i, j), (x1, x2) in edge_vars.items():
            if model[x1-1] > 0:
                count = 1
                if model[x2-1] > 0:
                    count = 2
                solution[(i, j)] = count
                used_literals.append(x1)
        if check_connectivity(solution, islands):
            solver.delete()
            return solution, islands, edges
        else:
            # Nếu nghiệm không liên thông, thêm mệnh đề chặn:
            # Mệnh đề này buộc rằng không được dùng đồng thời tất cả các x_e1 trong nghiệm hiện tại.
            solver.add_clause([-lit for lit in used_literals])
    
    solver.delete()
    return None, islands, edges

def print_solution(matrix, islands, edges, solution):
    """
    Hiển thị kết quả dưới dạng “bản đồ”:
      - Đảo được in số cầu cần có.
      - Cầu nối ngang được biểu diễn bằng '-' (1 cầu) hoặc '=' (2 cầu).
      - Cầu nối dọc được biểu diễn bằng '|' (1 cầu) hoặc '$' (2 cầu).
    """
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
            # Cầu ngang: điền các ô giữa
            r = r1
            for c in range(min(c1, c2)+1, max(c1, c2)):
                output[r][c] = '-' if count == 1 else '='
        elif c1 == c2:
            # Cầu dọc: điền các ô giữa
            c = c1
            for r in range(min(r1, r2)+1, max(r1, r2)):
                output[r][c] = '|' if count == 1 else '$'
    
    # In kết quả
    for row in output:
        print(' '.join(row))

if __name__ == '__main__':
    # Ví dụ puzzle từ bạn: 

    puzzle2= [
        [0 , 2 , 0 , 5 , 0 , 0 , 2],
        [0 , 0 , 0 , 0 , 0 , 0 , 0],
        [4 , 0 , 2 , 0 , 2 , 0 , 4],
        [0 , 0 , 0 , 0 , 0 , 0 , 0],
        [0 , 1 , 0 , 5 , 0 , 2 , 0],
        [0 , 0 , 0 , 0 , 0 , 0 , 0],
        [4 , 0 , 0 , 0 , 0 , 0 , 3],
    ]

    puzzle3= [
            [2,0,0,0,0,1,0],
            [0,0,2,0,0,0,3],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,2],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [2,0,2,0,0,0,0],]

    puzzle9_1 = [
            [0,0,3,0,2,0,0,0,1],
            [0,0,0,0,0,0,0,0,0],
            [1,0,5,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,2,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,3,0,0,0,0,2,0]
      ]
    puzzle9_2 = [
    [2,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,2,0,0,4],
    [0,0,0,0,0,0,0,0,0],
    [5,0,0,0,0,0,0,0,3],
    [0,0,0,0,0,0,0,0,0],
    [3,0,1,0,0,0,0,0,1],
    ]

    puzzle11_1 = [
    [2,0,0,0,0,0,0,2,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [4,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1],
    [0,0,3,0,0,0,0,6,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [3,0,0,0,0,0,0,4,0,0,3],

]
    puzzle11_2 = [
    [0,4,0,0,0,0,0,0,3,0,2],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,3,0,0,0,0,0,3,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,3,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,4,0,0,2,0,0,0,0],
    [0,3,0,0,0,0,0,0,0,2,0],

]
    puzzle13_1 = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,3,0,0,0,0,4,0,3,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [3,0,0,0,5,0,0,0,0,0,0,3,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [3,0,3,0,0,0,0,0,0,0,0,3,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [3,0,0,0,0,0,0,0,0,0,0,0,1],
]
    puzzle13_2 = [
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 5, 0, 0, 0, 0, 4, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

    puzzle15_1 = [
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1, 0, 0],
    [0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 0, 1],
]

    puzzle15_2 = [
    [1, 0, 5, 0, 0, 4, 0, 0, 0, 0, 2, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 4, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 3, 0, 0, 6, 0, 0, 0, 4, 0, 0, 6, 0, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 6, 0, 0, 0, 0, 4, 0, 5, 0, 4],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
    puzzle17_1 = [
    [3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 3],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 1, 0],
]

    puzzle17_2 = [
    [0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
    [3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
]



    puzzle20A = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0, 5, 0, 4, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 6, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0],
]
    puzzle20B = [
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0],
    [5, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 5, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 2, 0, 0, 0, 2, 0, 4, 0, 0, 0, 0, 0, 2, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
]

    solution, islands, edges = solve_hashiwokakero_lazy(puzzle17_2)
    if solution is not None:
        print("Giải pháp các cầu (edge: số cầu):")
        for edge, count in solution.items():
            print(f"Đảo {edge[0]} - Đảo {edge[1]}: {count} cầu")
        print("\nBản đồ kết quả:")
        print_solution(puzzle17_2, islands, edges, solution)
    else:
        print("Không tìm được lời giải cho puzzle.")
