from __future__ import annotations

import time
import math
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from heapq import heappush, heappop

from hxf_io import read_hxf, write_hxf, ORIENT_POINTY

# -----------------------------
# User-configurable input
# -----------------------------
HXF_PATH = Path('outputs/world_hexes_equal_earth.hxf')
CONTINENT = "south america"  # case-insensitive
OUTPUT_HXF = Path(f"outputs/{CONTINENT.lower()}_series_assignment.hxf")
RANDOM_SEED = 42
MAX_STALL_LOOPS = 50
DEBUG = True
BLOBINESS = 1

# Target global distribution (weights, not necessarily integers)
DISTRIBUTION = {
    "Eclipse": 13.2,
    "The Syndicate": 12.5,
    "Grumpy Old Bastards": 11.5,
    "Rose": 8.9,
    "Guardian": 7.1,
    "Event Horizon": 6.5,
    "The Knights Radiant": 4.5,
    "Yarr": 4.5,
    "Singularity": 4.3,
    "Citadel": 3.4,
    "The Fighting Pacifists": 2.9,
    "Myrmidons": 1.4,
    "The Immortals": 1.4,
}

# Physics growth parameters
SEED_JITTER = 1e-6  # small jitter for tie-breaking
PRESSURE_EPS = 1e-9  # numerical stability


# -----------------------------
# Apportionment helpers
# -----------------------------
def largest_remainder_apportion(weights: Dict[str, float], total: int) -> Dict[str, int]:
    keys = list(weights.keys())
    w = np.array([max(0.0, float(weights[k])) for k in keys], dtype=np.float64)
    if w.sum() <= 0:
        out = {k: 0 for k in keys}
        if keys:
            out[keys[0]] = total
        return out
    p = w / w.sum()
    ideal = p * total
    base = np.floor(ideal).astype(int)
    rem = ideal - base
    remain = total - int(base.sum())
    order = np.argsort(-rem)
    for j in order[:remain]:
        base[j] += 1
    return {k: int(v) for k, v in zip(keys, base.tolist())}


def split_counts_across_components(
    global_counts: Dict[str, int], comp_sizes: List[int]
) -> List[Dict[str, int]]:
    comps = len(comp_sizes)
    total_tiles = int(sum(comp_sizes))
    if total_tiles == 0 or comps == 0:
        return [{} for _ in range(comps)]

    per_comp_counts = [dict((k, 0) for k in global_counts.keys()) for _ in range(comps)]
    for series, g_count in global_counts.items():
        if g_count <= 0:
            continue
        ideal = np.array([g_count * (sz / total_tiles) for sz in comp_sizes], dtype=np.float64)
        base = np.floor(ideal).astype(int)
        rem = ideal - base
        remain = g_count - int(base.sum())
        order = np.argsort(-rem)
        alloc = base.copy()
        if remain > 0:
            alloc[order[:remain]] += 1
        for ci in range(comps):
            per_comp_counts[ci][series] = int(alloc[ci])

    # Per-component correction to match component size exactly
    for ci in range(comps):
        sz = int(comp_sizes[ci])
        sers = list(global_counts.keys())
        current = sum(per_comp_counts[ci][s] for s in sers)
        if current == sz:
            continue
        if current < sz:
            order = sorted(sers, key=lambda s: global_counts[s], reverse=True)
            k = 0
            while current < sz:
                per_comp_counts[ci][order[k % len(order)]] += 1
                k += 1
                current += 1
        else:
            order = sorted(sers, key=lambda s: global_counts[s])
            k = 0
            while current > sz:
                s = order[k % len(order)]
                if per_comp_counts[ci][s] > 0:
                    per_comp_counts[ci][s] -= 1
                    current -= 1
                k += 1
    return per_comp_counts


# -----------------------------
# Hex adjacency and components
# -----------------------------
def build_neighbor_graph(
    indices: np.ndarray,
    centers: np.ndarray,
    radius: float,
    orientation: int,
    tol_factor: float = 1e-3,
    debug: bool = False,
) -> Dict[int, List[int]]:
    if int(orientation) == int(ORIENT_POINTY):
        dx = math.sqrt(3.0) * radius
        dy = 1.5 * radius
        offsets = [
            (+dx, 0.0),
            (-dx, 0.0),
            (+dx / 2.0, +dy),
            (-dx / 2.0, +dy),
            (+dx / 2.0, -dy),
            (-dx / 2.0, -dy),
        ]
    else:
        dy = math.sqrt(3.0) * radius
        dx = 1.5 * radius
        offsets = [
            (0.0, +dy),
            (0.0, -dy),
            (+dx, +dy / 2.0),
            (+dx, -dy / 2.0),
            (-dx, +dy / 2.0),
            (-dx, -dy / 2.0),
        ]

    tol = max(1e-9, float(radius) * float(tol_factor))

    def key_xy(x: float, y: float) -> Tuple[int, int]:
        return (int(round(x / tol)), int(round(y / tol)))

    sub_idx = np.asarray(indices, dtype=np.int64)
    sub_centers = centers[sub_idx]
    lookup = {key_xy(float(x), float(y)): int(gidx) for gidx, (x, y) in zip(sub_idx, sub_centers)}
    adj: Dict[int, List[int]] = {int(i): [] for i in sub_idx}
    missing = 0
    for gidx, (x, y) in zip(sub_idx, sub_centers):
        x = float(x)
        y = float(y)
        for ox, oy in offsets:
            nb = lookup.get(key_xy(x + ox, y + oy))
            if nb is not None and nb in adj:
                adj[int(gidx)].append(nb)
            else:
                missing += 1
    if debug:
        degs = np.array([len(adj[i]) for i in sub_idx], dtype=int)
        print(f"[adj] nodes={len(sub_idx)} mean_deg={degs.mean():.2f} min={degs.min()} max={degs.max()} missing_lookups={missing}")
    return adj


def connected_components(nodes: List[int], adj: Dict[int, List[int]]) -> List[List[int]]:
    node_set = set(nodes)
    seen = set()
    comps: List[List[int]] = []
    for u in nodes:
        if u in seen:
            continue
        stack = [u]
        comp = []
        seen.add(u)
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in adj.get(v, []):
                if w in node_set and w not in seen:
                    seen.add(w)
                    stack.append(w)
        comps.append(comp)
    return comps


# -----------------------------
# Seeding
# -----------------------------
def farthest_point_seeds(
    comp_indices: List[int],
    centers: np.ndarray,
    k: int,
    rng: random.Random,
) -> List[int]:
    if k <= 0 or not comp_indices:
        return []
    idx = np.asarray(comp_indices, dtype=np.int64)
    P = centers[idx].astype(np.float64)
    centroid = P.mean(axis=0)
    d2_centroid = np.sum((P - centroid) ** 2, axis=1)
    # Pick farthest-from-centroid (periphery) as the first seed to avoid being boxed in
    first_local = int(np.argmax(d2_centroid))
    seeds = [int(idx[first_local])]
    if k == 1:
        return seeds
    d2min = np.sum((P - centers[seeds[0]]) ** 2, axis=1)
    for _ in range(1, k):
        next_local = int(np.argmax(d2min + rng.random() * 1e-9))
        next_seed = int(idx[next_local])
        seeds.append(next_seed)
        d2_new = np.sum((P - centers[next_seed]) ** 2, axis=1)
        d2min = np.minimum(d2min, d2_new)
    return seeds


def physics_growth_assign(
    comp_indices: List[int],
    adj: Dict[int, List[int]],
    centers: np.ndarray,
    local_counts: Dict[str, int],
    rng: random.Random,
) -> Dict[int, str]:
    # Series that need tiles (keep original order to minimize behavior changes)
    series_names = [s for s, c in local_counts.items() if c > 0]
    if not series_names:
        return {}

    # Local indexing for this component
    gl_idx = np.asarray(comp_indices, dtype=np.int64)
    n = int(gl_idx.size)
    loc_of = {int(g): i for i, g in enumerate(gl_idx)}

    # Local adjacency list (neighbors indexed 0..n-1)
    neighbors: List[List[int]] = []
    for g in gl_idx:
        lst = adj.get(int(g), [])
        neighbors.append([loc_of[v] for v in lst if v in loc_of])

    S = len(series_names)
    sid_of = {name: i for i, name in enumerate(series_names)}
    remain = np.array([int(local_counts[s]) for s in series_names], dtype=np.int32)
    size_by = np.zeros(S, dtype=np.int32)

    # Assigned series id per tile (-1 unassigned)
    assigned = np.full(n, -1, dtype=np.int16)

    # Seeds: farthest-point, map outermost to largest series (as in original)
    seeds_gl = farthest_point_seeds(comp_indices, centers, S, rng)
    seeds_loc = [loc_of[int(g)] for g in seeds_gl]

    centroid = centers[gl_idx].astype(np.float64).mean(axis=0)

    def d2_to_centroid_local(li: int) -> float:
        v = centers[int(gl_idx[li])].astype(np.float64) - centroid
        return float(v[0] * v[0] + v[1] * v[1])

    seeds_sorted_loc = sorted(seeds_loc, key=d2_to_centroid_local, reverse=True)
    series_sorted = sorted(series_names, key=lambda s: local_counts[s], reverse=True)

    seeds_by_sid = [-1] * S
    for seed, s in zip(seeds_sorted_loc, series_sorted):
        seeds_by_sid[sid_of[s]] = int(seed)

    # Per-series region tiles (for fast boundary building during rescue)
    region_tiles: List[Set[int]] = [set() for _ in range(S)]

    # Initialize seeds
    for sid in range(S):
        u = seeds_by_sid[sid]
        if assigned[u] == -1:
            assigned[u] = sid
            remain[sid] -= 1
            size_by[sid] += 1
            region_tiles[sid].add(u)

    # Per-series frontier heap: (priority:int, tie:int, u:int)
    frontiers: List[List[Tuple[int, int, int]]] = [[] for _ in range(S)]
    # Frontier membership mask (S x n)
    frontier_mask: List[bytearray] = [bytearray(n) for _ in range(S)]
    # Cheap per-series tiebreaker counter (faster than rng.random in tight loop)
    push_counter = [0] * S
    inv_blob = 1.0 / BLOBINESS if BLOBINESS != 0 else 1.0  # safe

    def nb_same_neighbors(u: int, sid: int) -> int:
        cnt = 0
        for v in neighbors[u]:
            if assigned[v] == sid:
                cnt += 1
        return cnt

    def push_frontier_for(sid: int, u: int):
        if assigned[u] != -1:
            return
        fm = frontier_mask[sid]
        if fm[u]:
            return
        b = nb_same_neighbors(u, sid)
        if b < 0:
            b = 0
        elif b > 6:
            b = 6
        # Priority: -b (prefer more same-labeled neighbors), tie by monotonic counter
        cnt = push_counter[sid]
        push_counter[sid] = cnt + 1
        heappush(frontiers[sid], (-b, cnt, u))
        fm[u] = 1

    def rebuild_frontier_for(sid: int):
        frontier_mask[sid] = bytearray(n)
        heap = frontiers[sid]
        heap.clear()
        # Add all unassigned neighbors of current region
        for u in region_tiles[sid]:
            for v in neighbors[u]:
                if assigned[v] == -1:
                    push_frontier_for(sid, v)

    # Seed frontiers from seed neighbors
    for sid in range(S):
        u = seeds_by_sid[sid]
        for v in neighbors[u]:
            if assigned[v] == -1:
                push_frontier_for(sid, v)

    def expand_one(sid: int) -> Optional[int]:
        heap = frontiers[sid]
        fm = frontier_mask[sid]
        while heap:
            _, _, u = heappop(heap)
            # Skip stale entries
            if assigned[u] != -1:
                continue
            # Claim tile
            assigned[u] = sid
            fm[u] = 0  # clean mask for this sid
            remain[sid] -= 1
            size_by[sid] += 1
            region_tiles[sid].add(u)
            # Offer neighbors
            for w in neighbors[u]:
                if assigned[w] == -1:
                    push_frontier_for(sid, w)
            return u
        return None

    def is_peelable(u: int, sid: int) -> bool:
        # A tile is peelable if it has <=1 same-labeled neighbors
        return nb_same_neighbors(u, sid) <= 1

    def connected_after_removal_fast(u: int, sid: int) -> bool:
        if size_by[sid] <= 1:
            return True
        # Find any neighbor with same label
        start = -1
        for v in neighbors[u]:
            if assigned[v] == sid:
                start = v
                break
        if start < 0:
            return True
        target = size_by[sid] - 1
        seen = bytearray(n)
        seen[u] = 1
        seen[start] = 1
        dq = deque([start])
        count = 1
        while dq:
            x = dq.popleft()
            for y in neighbors[x]:
                if seen[y] or assigned[y] != sid:
                    continue
                seen[y] = 1
                dq.append(y)
                count += 1
                if count >= target:
                    return True
        return False

    # Progress diagnostics
    total_target = int(sum(local_counts.values()))
    start_ts = time.time()
    last_log_ts = start_ts
    MAX_BOUNDARY_SAMPLE = 8000
    MAX_CONNECTIVITY_CHECKS = 1000
    stall_loops = 0
    prev_remaining_total = int(remain.sum())

    def log_progress(force: bool = False, note: str = ""):
        nonlocal last_log_ts
        if not DEBUG:
            return
        now = time.time()
        if not force and (now - last_log_ts) < 5.0:  # log less often for speed
            return
        remaining_total = int(remain.sum())
        zero_frontiers = sum(1 for sid in range(S) if remain[sid] > 0 and len(frontiers[sid]) == 0)
        pct = 100.0 * (int((assigned != -1).sum()) / max(1, total_target))
        msg = f"[grow] {int((assigned != -1).sum())}/{total_target} ({pct:.1f}%) remain={remaining_total} zero_frontiers={zero_frontiers} elapsed={now - start_ts:.1f}s"
        if note:
            msg += f" | {note}"
        print(msg)
        tail = sorted([(series_names[sid], int(remain[sid]), len(frontiers[sid])) for sid in range(S)],
                      key=lambda x: x[1], reverse=True)[:5]
        print("[grow] top remain:", ", ".join(f"{s}:{r}/F{f}" for s, r, f in tail))
        last_log_ts = now

    # Main growth loop
    while True:
        total_remaining = int(remain.sum())
        if total_remaining <= 0:
            break

        # Choose by pressure
        best_sid = -1
        best_pressure = -1.0
        for sid in range(S):
            r = int(remain[sid])
            if r <= 0:
                continue
            fsize = len(frontiers[sid])  # lazy-stale, matches original semantics
            if fsize == 0:
                continue
            if BLOBINESS == 1:
                denom = float(size_by[sid]) if size_by[sid] > 0 else PRESSURE_EPS
            else:
                denom = (size_by[sid] ** inv_blob) if size_by[sid] > 0 else PRESSURE_EPS
            pressure = (float(r) / float(fsize) + PRESSURE_EPS) / denom
            # very small jitter (deterministic) to break ties cheaply
            pressure += (push_counter[sid] & 0xFFFF) * 1e-12
            if pressure > best_pressure:
                best_pressure = pressure
                best_sid = sid

        progressed_assign = False

        if best_sid >= 0:
            u = expand_one(best_sid)
            if u is None:
                rebuild_frontier_for(best_sid)
                u = expand_one(best_sid)
            if u is not None:
                progressed_assign = True
        else:
            # Rescue: peel from neighbors to unblock zero-frontier series
            rescued = False
            order = sorted(range(S), key=lambda sid: int(remain[sid]), reverse=True)
            for sid in order:
                if remain[sid] <= 0:
                    continue
                # Build boundary list of this sid (iterate only its region tiles)
                boundary_all: List[int] = []
                for u in region_tiles[sid]:
                    for v in neighbors[u]:
                        if assigned[v] != sid:
                            boundary_all.append(u)
                            break
                if not boundary_all:
                    continue
                # Sample boundary if too large
                if len(boundary_all) > MAX_BOUNDARY_SAMPLE:
                    boundary = rng.sample(boundary_all, MAX_BOUNDARY_SAMPLE)
                else:
                    boundary = boundary_all
                    rng.shuffle(boundary)

                conn_checks = 0
                for u in boundary:
                    for v in neighbors[u]:
                        lab_v = int(assigned[v])
                        if lab_v < 0 or lab_v == sid:
                            continue
                        peel_ok = is_peelable(v, lab_v)
                        conn_ok = False
                        if not peel_ok and conn_checks < MAX_CONNECTIVITY_CHECKS:
                            conn_ok = connected_after_removal_fast(v, lab_v)
                            conn_checks += 1
                        if not (peel_ok or conn_ok):
                            continue
                        # Relabel v -> sid
                        assigned[v] = sid
                        remain[sid] -= 1
                        remain[lab_v] += 1
                        size_by[sid] += 1
                        size_by[lab_v] -= 1
                        region_tiles[sid].add(v)
                        if v in region_tiles[lab_v]:
                            region_tiles[lab_v].remove(v)
                        # Add v's unassigned neighbors to sid frontier
                        for w in neighbors[v]:
                            if assigned[w] == -1:
                                push_frontier_for(sid, w)
                        # Try to immediately expand donor to make net progress
                        if remain[lab_v] > 0:
                            u2 = expand_one(lab_v)
                            if u2 is None:
                                rebuild_frontier_for(lab_v)
                                u2 = expand_one(lab_v)
                            if u2 is not None:
                                progressed_assign = True
                        else:
                            progressed_assign = True
                        rescued = True
                        break
                    if rescued:
                        break
            if not rescued:
                log_progress(force=True, note="rescue stalled")
                raise RuntimeError("Rescue failed: no peelable boundary found to satisfy remaining quotas.")

        cur_remaining_total = int(remain.sum())
        if progressed_assign or cur_remaining_total < prev_remaining_total:
            stall_loops = 0
        else:
            stall_loops += 1

        if DEBUG:
            log_progress(force=not progressed_assign, note=("no-progress" if not progressed_assign else ""))

        if stall_loops > MAX_STALL_LOOPS:
            for sid in range(S):
                if remain[sid] > 0:
                    rebuild_frontier_for(sid)
            log_progress(force=True, note="rebuilt all frontiers")
            stall_loops = 0

        prev_remaining_total = cur_remaining_total

    # Map back to globals and names
    out: Dict[int, str] = {}
    for li in range(n):
        sid = int(assigned[li])
        if sid >= 0:
            out[int(gl_idx[li])] = series_names[sid]
    return out


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    rng = random.Random(RANDOM_SEED)

    meta = read_hxf(HXF_PATH)
    centers = np.asarray(meta["centers"], dtype=np.float32)
    radius = float(meta["radius_m"])
    orientation = int(meta["orientation"])
    epsg = int(meta["epsg"])
    class_mask = meta.get("class_is_land")
    labels = meta.get("continent_labels")
    codes = meta.get("continent_codes")

    if labels is None or codes is None:
        raise RuntimeError("Continent metadata not found in HXF. Regenerate with add_meta=True.")

    label_by_code = {i: lab for i, lab in enumerate(labels)}
    cont_names = np.array([label_by_code[int(c)] for c in codes], dtype=object)
    cont_names_lc = np.char.lower(cont_names.astype(str))

    land_mask = np.ones(len(centers), dtype=bool) if class_mask is None else class_mask.astype(bool)
    want_cont = cont_names_lc == CONTINENT.lower()

    idx_all = np.arange(len(centers), dtype=np.int64)
    idx_continent = idx_all[land_mask & want_cont]
    if len(idx_continent) == 0:
        raise RuntimeError(f"No tiles found for continent={CONTINENT!r}.")

    # Neighbor graph restricted to continent tiles
    adj = build_neighbor_graph(idx_continent, centers, radius, orientation, tol_factor=1e-3, debug=DEBUG)

    # Connected components (islands)
    comps = connected_components(idx_continent.tolist(), adj)
    comps = [c for c in comps if c]
    comp_sizes = [len(c) for c in comps]
    if DEBUG:
        print(f"[info] continent components: {len(comps)} sizes={comp_sizes[:10]}{'...' if len(comp_sizes)>10 else ''}")

    # Apportion global counts to components
    total_tiles = int(sum(comp_sizes))
    global_counts = largest_remainder_apportion(DISTRIBUTION, total_tiles)
    per_comp_counts = split_counts_across_components(global_counts, comp_sizes)

    # Assign per component via physics-like contiguous growth
    assigned_series: Dict[int, str] = {}
    for ci, (comp_indices, local_counts) in enumerate(zip(comps, per_comp_counts)):
        # filter out zero entries and skip empty work
        local_counts = {s: c for s, c in local_counts.items() if c > 0}
        if sum(local_counts.values()) == 0:
            continue
        if DEBUG:
            print(f"[comp {ci}] size={len(comp_indices)} series={len(local_counts)}")
        # create a component-local RNG seeded from the global RNG to keep reproducibility
        comp_seed = rng.randrange(0, 2**32)
        comp_rng = random.Random(comp_seed)
        local_assigned = physics_growth_assign(
            comp_indices=comp_indices,
            adj=adj,
            centers=centers,
            local_counts=local_counts,
            rng=comp_rng,
        )
        assigned_series.update(local_assigned)

    # Sanity: ensure all continent tiles are assigned (should be)
    unassigned = [int(i) for i in idx_continent if int(i) not in assigned_series]
    if unassigned:
        # Assign leftovers to largest global series
        largest = max(global_counts.items(), key=lambda kv: kv[1])[0]
        for i in unassigned:
            assigned_series[i] = largest
        if DEBUG:
            print(f"[post] unassigned tiles: {len(unassigned)} -> assigned to '{largest}'")

    # Prepare output subset and labels (preserve original ordering of idx_continent)
    centers_out = centers[idx_continent]
    series_per_idx = [assigned_series[int(i)] for i in idx_continent]
    class_is_land_out = np.ones(len(idx_continent), dtype=bool)

    OUTPUT_HXF.parent.mkdir(parents=True, exist_ok=True)
    write_hxf(
        OUTPUT_HXF,
        centers_out,
        epsg=epsg,
        radius_m=radius,
        orientation=orientation,
        class_is_land=class_is_land_out,
        continent_names=series_per_idx,  # store series labels as categories
    )

    # Summary
    final_counts = defaultdict(int)
    for s in series_per_idx:
        final_counts[s] += 1

    print(f"\nContinent: {CONTINENT} tiles={len(idx_continent)} components={len(comps)}")
    print("Target counts (global):", dict(global_counts))
    print("Final counts:", dict(final_counts))
    print(f"Wrote assignment HXF to: {OUTPUT_HXF}")


if __name__ == "__main__":
    startMs = time.time()
    main()
    diff = time.time() - startMs
    print(f"\nDone. Elapsed time: {diff:.1f} seconds")