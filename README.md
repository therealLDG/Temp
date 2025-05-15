# 1. Tic-Tac-Toe (Human vs AI) using Minimax
    import random

    def tic_tac_toe():
        board = [' '] * 9
        human, ai = 'X', 'O'

    def print_board():
        for i in range(0, 9, 3):
            print('|'.join(board[i:i+3]))
            if i < 6:
                print('-' * 5)

    def win(b, p):
        # Check all winning triples
        wins = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]
        return any(b[i] == b[j] == b[k] == p for i, j, k in wins)

    def minimax(b, player):
        if win(b, human):
            return {'score': -1}
        if win(b, ai):
            return {'score': 1}
        if ' ' not in b:
            return {'score': 0}

        moves = []
        for idx in range(9):
            if b[idx] == ' ':
                b[idx] = player
                result = minimax(b, ai if player == human else human)
                moves.append({'index': idx, 'score': result['score']})
                b[idx] = ' '

        # AI maximizes score, human minimizes
        if player == ai:
            return max(moves, key=lambda m: m['score'])
        else:
            return min(moves, key=lambda m: m['score'])

    turn = human
    for _ in range(9):
        print_board()
        if turn == human:
            try:
                idx = int(input(f"Human ({human}) move (0-8): "))
            except ValueError:
                print("Please enter a number 0-8.")
                continue
            if idx < 0 or idx > 8 or board[idx] != ' ':
                print("Invalid move. Try again.")
                continue
            board[idx] = human
        else:
            move = minimax(board, ai)
            board[move['index']] = ai
            print(f"AI ({ai}) chooses {move['index']}")

        # Check for a win after each move
        if win(board, turn):
            print_board()
            print(f"{turn} wins!")
            return

        # Switch turn
        turn = ai if turn == human else human

    print_board()
    print("It's a draw!")


    if __name__ == '__main__':
        tic_tac_toe()


# 2. Water Jug Problem (4L & 3L jugs -> measure 2L)
    from collections import deque

    def water_jug():
        cap = (4, 3)
        target = 2
        visited = set()
        parent = {}

    def neighbors(state):
        a, b = state
        res = []
        # fill A, fill B, empty A, empty B
        res.extend([(cap[0], b), (a, cap[1]), (0, b), (a, 0)])
        # pour A->B
        pour = min(a, cap[1] - b)
        res.append((a - pour, b + pour))
        # pour B->A
        pour = min(b, cap[0] - a)
        res.append((a + pour, b - pour))
        return res

    queue = deque([(0, 0)])
    visited.add((0, 0))
    while queue:
        state = queue.popleft()
        if target in state:
            path = []
            while state in parent:
                path.append(state)
                state = parent[state]
            path.append((0, 0))
            path.reverse()
            print("Solution path:", path)
            return
        for nxt in neighbors(state):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = state
                queue.append(nxt)
    print("No solution")


    if __name__ == '__main__':
        water_jug()


# 3. Breadth First Search (BFS)
    def bfs(graph, start):
        visited = {start}
        queue = [start]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for nbr in graph.get(node, []):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        return order


    if __name__ == '__main__':
        sample_graph = {'A': ['B', 'C'], 'B': ['D'], 'C': ['E'], 'D': [], 'E': []}
        print("BFS order:", bfs(sample_graph, 'A'))


# 4. Travelling Salesman via Hill Climbing
    import random

    def tsp_hill_climbing(dist):
        nodes = list(dist.keys())
        current = nodes[:]
        random.shuffle(current)
        best = current[:]

    def cost(path):
        return sum(dist[path[i]][path[(i+1) % len(path)]] for i in range(len(path)))

    best_cost = cost(best)
    improved = True
    while improved:
        improved = False
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                neighbor = best[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                c = cost(neighbor)
                if c < best_cost:
                    best_cost = c
                    best = neighbor
                    improved = True
    return best, best_cost


    if __name__ == '__main__':
        distances = {
            'A': {'B': 10, 'C': 15, 'D': 20},
            'B': {'A': 10, 'C': 35, 'D': 25},
            'C': {'A': 15, 'B': 35, 'D': 30},
            'D': {'A': 20, 'B': 25, 'C': 30}
        }
        path, cost_val = tsp_hill_climbing(distances)
        print("Path:", path, "Cost:", cost_val)


# 5. Simulated Annealing for TSP
    import math

    def simulated_annealing(dist, T0=10000, alpha=0.995, stopping_T=1e-3):
        nodes = list(dist.keys())
        current = nodes[:]
        random.shuffle(current)
        best = current[:]

    def cost(path):
        return sum(dist[path[i]][path[(i+1) % len(path)]] for i in range(len(path)))

    current_cost = cost(current)
    best_cost = current_cost
    T = T0
    while T > stopping_T:
        i, j = random.sample(range(len(nodes)), 2)
        neighbor = current[:]
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        c = cost(neighbor)
        if c < current_cost or random.random() < math.exp((current_cost - c) / T):
            current, current_cost = neighbor, c
            if c < best_cost:
                best, best_cost = neighbor, c
        T *= alpha
    return best, best_cost


    if __name__ == '__main__':
        sample_dist = {'A': {'B': 5, 'C': 8}, 'B': {'A': 5, 'C': 3}, 'C': {'A': 8, 'B': 3}}
        p, c = simulated_annealing(sample_dist)
        print("SA Path:", p, "Cost:", c)


# 6. Latin Square (n×n CSP) via Backtracking
    def latin_square(n):
        grid = [[0] * n for _ in range(n)]

    def ok(r, c, val):
        return all(grid[r][j] != val for j in range(n)) and all(grid[i][c] != val for i in range(n))

    def solve(pos=0):
        if pos == n * n:
            return True
        r, c = divmod(pos, n)
        for val in range(1, n + 1):
            if ok(r, c, val):
                grid[r][c] = val
                if solve(pos + 1):
                    return True
                grid[r][c] = 0
        return False

    if solve():
        for row in grid:
            print(row)
    else:
        print("No solution")


    if __name__ == '__main__':
        latin_square(5)


# 7. Unification Algorithm (Robinson's)
    def unify(x, y, s=None):
        if s is None:
            s = {}
        if x == y:
            return s
        if is_var(x):
            return unify_var(x, y, s)
        if is_var(y):
            return unify_var(y, x, s)
        if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for xi, yi in zip(x, y):
                s = unify(apply(s, xi), apply(s, yi), s)
                if s is None:
                    return None
            return s
        return None

    def is_var(x):
        return isinstance(x, str) and x[0].islower()

    def apply(s, x):
        if is_var(x) and x in s:
            return apply(s, s[x])
        if isinstance(x, list):
            return [apply(s, xi) for xi in x]
        return x

    def unify_var(v, x, s):
        if v in s:
            return unify(s[v], x, s)
        if is_var(x) and x in s:
            return unify(v, s[x], s)
        if occurs(v, x):
            return None
        s2 = s.copy()
        s2[v] = x
        return s2

    def occurs(v, x):
        if v == x:
            return True
        if isinstance(x, list):
            return any(occurs(v, xi) for xi in x)
        return False

    if __name__ == '__main__':
        print(unify(['f', 'X'], ['f', 'a']))


# 8. Q-Learning (Grid World)
    def q_learning(n=4, eps=1000, alpha=0.1, gamma=0.9, eps_greedy=0.2):
        import random
        Q = {(i, j):{a: 0 for a in ['up', 'down', 'left', 'right']} for i in range(n) for j in range(n)}
        def reward(s): return 1 if s == (n-1, n-1) else 0
        def step(s, a):
            i, j = s
            moves = {
                'up': (max(i-1, 0), j),
                'down': (min(i+1, n-1), j),
                'left': (i, max(j-1, 0)),
                'right': (i, min(j+1, n-1))
            }
            return moves[a]
        for _ in range(eps):
            s = (0, 0)
            while s != (n-1, n-1):
                if random.random() < eps_greedy:
                    a = random.choice(list(Q[s].keys()))
                else:
                    a = max(Q[s], key=Q[s].get)
                ns = step(s, a)
                r = reward(ns)
                Q[s][a] += alpha * (r + gamma * max(Q[ns].values()) - Q[s][a])
                s = ns
        for state, acts in Q.items():
            print(state, acts)
    
    if __name__ == '__main__':
        q_learning()
