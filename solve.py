import torch, heapq, time, random
from cube2x2 import Cube2x2
from dca_model import DCAModel

device = "cpu"

# Load trained DCA model (updated filename)
model = DCAModel().to(device)
model.load_state_dict(torch.load("dca.bin", map_location=device))
model.eval()


def solve(depth=6):
    print("=== SOLVER START ===")
    print(f"Scramble depth: {depth}")

    # Scramble the cube
    cube = Cube2x2()
    scramble_moves = []
    for _ in range(depth):
        m = random.choice(Cube2x2.MOVES)
        cube.move(m)
        scramble_moves.append(m)

    print("Scramble sequence:", scramble_moves)
    print()

    # Priority queue requires tie-break counter
    counter = 0
    pq = [(0, counter, cube.clone(), [])]
    visited = set()

    start_time = time.time()

    while pq:
        priority, _, cube, path = heapq.heappop(pq)

        key = (tuple(cube.perm), tuple(cube.orient))
        if key in visited:
            continue
        visited.add(key)

        # If cube solved — print results
        if cube.is_solved():
            end = time.time()

            print("=== SOLVED ===")
            print(f"Moves used: {len(path)}")
            print("Solution path:", path)
            print(f"Time taken: {end - start_time:.4f} seconds")
            print()

            return path

        # Expand neighbors
        for m in Cube2x2.MOVES:
            nxt = cube.clone()
            nxt.move(m)

            inp = torch.tensor(nxt.to_onehot(), dtype=torch.float32)
            with torch.no_grad():
                pred = model(inp).item()

            counter += 1
            heapq.heappush(pq, (pred, counter, nxt, path + [m]))

    print("=== UNSOLVABLE (unexpected) ===")
    return None


# Run solver
solve(6)
