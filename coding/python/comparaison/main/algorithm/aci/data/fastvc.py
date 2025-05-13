import torch
import time
import random

def construct_vc(A):
    N = A.shape[0]
    degrees = A.sum(dim=1)
    C = set()
    covered = torch.zeros_like(A)

    # Extension phase
    for u in range(N):
        for v in range(N):
            if A[u, v] == 1 and covered[u, v] == 0:
                # Edge (u,v) not covered
                if degrees[u] >= degrees[v]:
                    chosen = u
                else:
                    chosen = v
                C.add(chosen)
                covered[chosen, :] = 1
                covered[:, chosen] = 1

    # Compute loss
    loss = torch.zeros(N)
    for u in range(N):
        for v in range(N):
            if A[u, v] == 1:
                if u in C and v not in C:
                    loss[u] += 1
                elif v in C and u not in C:
                    loss[v] += 1

    # Shrinking phase
    changed = True
    while changed:
        changed = False
        for v in list(C):
            if loss[v] == 0:
                C.remove(v)
                for u in range(N):
                    if A[u, v] == 1 and u in C:
                        loss[u] += 1
                changed = True

    return C

def bms_selection(candidates, loss, k=50):
    best = random.choice(candidates)
    for _ in range(k - 1):
        r = random.choice(candidates)
        if loss[r] < loss[best]:
            best = r
    return best

def fastvc(A, cutoff=10, max_iter=30):
    N = A.shape[0]
    best_cover = None
    best_size = N + 1

    start_time = time.time()

    for _ in range(5):  # 5 runs as in Lazzarinetti
        C = construct_vc(A)
        loss = torch.zeros(N)
        gain = torch.zeros(N)
        in_cover = torch.zeros(N, dtype=torch.bool)
        for v in C:
            in_cover[v] = True

        # Init gain
        for u in range(N):
            if not in_cover[u]:
                for v in range(N):
                    if A[u, v] == 1 and in_cover[v] == 0:
                        gain[u] += 1

        elapsed = 0
        it = 0
        while elapsed < cutoff and it < max_iter:
            it += 1
            uncovered = []
            for u in range(N):
                if not in_cover[u]:
                    for v in range(N):
                        if A[u, v] == 1 and not in_cover[v]:
                            uncovered.append((u, v))
            if not uncovered:
                if len(C) < best_size:
                    best_cover = C.copy()
                    best_size = len(C)
                # remove vertex with min loss
                candidates = list(C)
                losses = {v: 0 for v in candidates}
                for u in range(N):
                    if in_cover[u]:
                        for v in range(N):
                            if A[u, v] == 1 and in_cover[v]:
                                losses[u] += 1
                min_loss = min(losses.values())
                for v in candidates:
                    if losses[v] == min_loss:
                        C.remove(v)
                        in_cover[v] = False
                        break
                continue

            # remove vertex with BMS
            candidates = list(C)
            losses = {v: 0 for v in candidates}
            for u in range(N):
                if in_cover[u]:
                    for v in range(N):
                        if A[u, v] == 1 and in_cover[v]:
                            losses[u] += 1
            u = bms_selection(candidates, losses)
            C.remove(u)
            in_cover[u] = False

            # update gain
            gain = torch.zeros(N)
            for x in range(N):
                if not in_cover[x]:
                    for y in range(N):
                        if A[x, y] == 1 and not in_cover[y]:
                            gain[x] += 1

            # pick uncovered edge and vertex with highest gain
            e = random.choice(uncovered)
            u, v = e
            if gain[u] > gain[v]:
                to_add = u
            elif gain[v] > gain[u]:
                to_add = v
            else:
                to_add = u if random.random() < 0.5 else v

            C.add(to_add)
            in_cover[to_add] = True

            elapsed = time.time() - start_time

    y = torch.zeros(N)
    y[list(best_cover)] = 1.0
    return y
