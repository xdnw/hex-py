from __future__ import annotations

import math
import random
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from hxf_io import read_hxf, write_hxf, ORIENT_POINTY

# -----------------------------
# User-configurable mock input
# -----------------------------
HXF_PATH = Path("outputs/world_hexes_equal_earth.hxf")
CONTINENT = "south america"  # lower/upper will be normalized
OUTPUT_HXF = Path(f"outputs/{CONTINENT.lower()}_series_assignment.hxf")
RANDOM_SEED = 42
DEBUG = True

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

# -----------------------------
# Helpers: apportionment
# -----------------------------
def largest_remainder_apportion(weights: Dict[str, float], total: int) -> Dict[str, int]:
    """Hamilton method to convert weights to integers summing to total."""
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
    """Split per-series totals across components proportional to component size."""
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
# Helpers: adjacency on hex grid
# -----------------------------
def build_neighbor_graph(
    indices: np.ndarray,
    centers: np.ndarray,
    radius: float,
    orientation: int,
    tol_factor: float = 1e-3,
    debug: bool = False,
) -> Dict[int, List[int]]:
    """
    Adjacency among 'indices' using hex neighbor offsets.
    Uses a tolerance-based hash to be robust to floating noise.

    tol_factor: rounding tolerance as a fraction of radius (default 1e-3).
    """
    # Geometry by orientation
    if int(orientation) == int(ORIENT_POINTY):
        # pointy-top
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
        # flat-top
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

    # Tolerance grid
    tol = max(1e-9, float(radius) * float(tol_factor))

    def key_xy(x: float, y: float) -> Tuple[int, int]:
        # Quantize by tol to absorb small numeric discrepancies
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
        # Spot-check some nodes
        sample = sub_idx[: min(10, len(sub_idx))]
        for i in sample:
            print(f"[adj] idx={int(i)} deg={len(adj[int(i)])}")

    return adj


def connected_components(nodes: List[int], adj: Dict[int, List[int]]) -> List[List[int]]:
    """Connected components among given nodes via adjacency."""
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
# Region growing per component
# -----------------------------
def farthest_point_seeds(comp_indices: List[int], centers: np.ndarray, k: int, rng: random.Random) -> List[int]:
    """Pick k seeds via farthest-point sampling."""
    if k <= 0:
        return []
    if k == 1:
        return [rng.choice(comp_indices)]
    pts = centers[comp_indices]
    seeds = [rng.choice(comp_indices)]
    seeds_pts = [centers[seeds[0]]]
    for _ in range(1, k):
        d2min = []
        for i, p in zip(comp_indices, pts):
            d2 = min(((p[0] - sp[0]) ** 2 + (p[1] - sp[1]) ** 2) for sp in seeds_pts)
            d2min.append((d2, i))
        _, far_idx = max(d2min, key=lambda t: t[0])
        seeds.append(int(far_idx))
        seeds_pts.append(centers[far_idx])
    return seeds


def region_grow_component(
    comp_indices: List[int],
    adj: Dict[int, List[int]],
    series_names: List[str],
    local_counts: Dict[str, int],
    centers: np.ndarray,
    rng: random.Random,
    debug: bool = False,
) -> Dict[int, str]:
    """
    Support-weighted multi-source growth with:
    - Majority-wins guard: if another series has strictly higher support for a tile and has budget,
      do not claim it. This fills holes and prevents single-hex enclaves.
    - Batched high-support claims with randomized series order to avoid striping.
    - Connectivity-preserving completion: redistribute leftover budgets from stuck series to those
      with frontier; finally assign remaining tiles by max-support while borrowing budget from
      stuck series. No nearest-seed non-adjacent fallback (prevents disconnected islands).
    """
    # Active series and budgets
    active = [(s, int(c)) for s, c in local_counts.items() if c > 0]
    if not active:
        return {}
    series_order = [s for s, _ in active]
    budgets = {s: c for s, c in active}
    k = len(series_order)

    # Tunables
    SAMPLE_K = 12          # candidates sampled from a bucket for scoring
    MAX_CLAIMS_HI = 4      # per-turn consecutive claims when support >= 2
    JITTER = 1e-6          # tiny tie-breaker
    RELAXED_GAP = 1        # on relaxed round, allow sup == max_sup - RELAXED_GAP

    # Seeds via farthest-point sampling
    seed_candidates = farthest_point_seeds(comp_indices, centers, k, rng)
    seeds_by_series = {s: seed_candidates[i] for i, s in enumerate(series_order)}

    assigned: Dict[int, str] = {}
    for s in series_order:
        v = seeds_by_series[s]
        assigned[v] = s
        budgets[s] -= 1

    if debug:
        print("[grow] seeds:")
        for s in series_order:
            v = seeds_by_series[s]
            cx, cy = centers[v]
            print(f"  {s}: idx={v} at ({float(cx):.3f}, {float(cy):.3f}) budget={budgets[s]}")

    comp_set = set(comp_indices)

    # Per-series candidate buckets by support (0..6) and live support counts
    candidate_buckets: Dict[str, List[set]] = {s: [set() for _ in range(7)] for s in series_order}
    support_counts: Dict[str, defaultdict] = {s: defaultdict(int) for s in series_order}

    # Initialize candidates from seeds
    for s in series_order:
        v = seeds_by_series[s]
        for nb in adj.get(v, []):
            if nb in comp_set and nb not in assigned:
                sc = support_counts[s][nb] + 1
                support_counts[s][nb] = sc
                candidate_buckets[s][sc].add(nb)

    def rebucket_if_drifted(s: str, nb: int, expected_sup: int) -> int | None:
        """Ensure nb is in the bucket matching its current support for series s."""
        live = support_counts[s].get(nb, 0)
        if live != expected_sup:
            if 0 <= expected_sup <= 6:
                candidate_buckets[s][expected_sup].discard(nb)
            if 0 <= live <= 6:
                candidate_buckets[s][live].add(nb)
            return None
        return live

    def global_max_support(nb: int) -> tuple[int, str]:
        """Max support across all series for tile nb."""
        best_sup, best_s = -1, None
        for s in series_order:
            sup = support_counts[s].get(nb, 0)
            if sup > best_sup:
                best_sup, best_s = sup, s
        return best_sup, best_s  # (support, series)

    def can_claim(s: str, nb: int, sup: int, relaxed: bool) -> bool:
        """Majority-wins guard with optional relaxation."""
        max_sup, max_s = global_max_support(nb)
        if max_s is None:
            return True
        # If someone else has strictly higher support and still has budget, we should not claim.
        if s != max_s and sup < max_sup and budgets.get(max_s, 0) > 0:
            # In relaxed mode allow small gap (e.g., sup == max_sup - 1) for high-support tiles only.
            if relaxed and sup >= max(2, max_sup - RELAXED_GAP):
                return True
            return False
        return True

    def score_candidates(s: str, sup: int, sample: List[int]) -> tuple[int | None, int | None, float]:
        """Score candidates in a bucket and pick best."""
        best_nb, best_sup, best_score = None, None, -1.0
        for nb in sample:
            if nb in assigned:
                continue
            live = rebucket_if_drifted(s, nb, sup)
            if live is None:
                continue
            # Cohesion bonus: sum of supports of nb's unassigned neighbors for this series
            bonus = 0
            for w in adj.get(nb, []):
                if w in comp_set and w not in assigned:
                    bonus += support_counts[s].get(w, 0)
            score = live + 0.1 * bonus + rng.random() * JITTER
            if score > best_score:
                best_nb, best_sup, best_score = nb, live, score
        if best_nb is not None:
            candidate_buckets[s][best_sup].discard(best_nb)
        return best_nb, best_sup, best_score

    def pick_best(s: str, min_sup: int, relaxed: bool) -> Tuple[int | None, int | None, float]:
        """
        Pick the best candidate for series s with support >= min_sup,
        respecting majority-wins guard (with optional relaxed mode).
        """
        for sup in range(6, min_sup - 1, -1):
            bucket = candidate_buckets[s][sup]
            if not bucket:
                continue
            # Random sample to avoid scanning large sets
            if len(bucket) <= SAMPLE_K:
                sample = list(bucket)
            else:
                # Reservoir-like sampling indices from a set
                idxs = set(rng.sample(range(len(bucket)), SAMPLE_K))
                sample = []
                for i, v in enumerate(bucket):
                    if i in idxs:
                        sample.append(v)
                        if len(sample) == SAMPLE_K:
                            break
            nb, live_sup, score = score_candidates(s, sup, sample)
            if nb is None:
                continue
            # Majority guard
            if not can_claim(s, nb, live_sup, relaxed=relaxed):
                continue
            return nb, live_sup, score
        return None, None, 0.0

    def series_frontier_size(s: str) -> int:
        return sum(len(bucket) for bucket in candidate_buckets[s])

    def redistribute_leftovers():
        """Move leftover budgets from stuck series (no frontier) to those with frontier."""
        stuck = [s for s in series_order if budgets[s] > 0 and series_frontier_size(s) == 0]
        if not stuck:
            return False
        give = sum(budgets[s] for s in stuck)
        if give <= 0:
            return False
        for s in stuck:
            if debug:
                print(f"[rebalance] {s} stuck; releasing {budgets[s]} tiles")
            budgets[s] = 0
        receivers = [s for s in series_order if series_frontier_size(s) > 0]
        if not receivers:
            return False
        # Allocate by frontier size (larger frontiers get more)
        weights = {s: float(series_frontier_size(s)) for s in receivers}
        total_w = sum(weights.values())
        if total_w <= 0:
            # fallback: equal split
            share = give // len(receivers)
            rem = give - share * len(receivers)
            for s in receivers:
                budgets[s] += share
            for s in receivers[:rem]:
                budgets[s] += 1
        else:
            # Largest remainder apportionment
            ideals = {s: weights[s] / total_w * give for s in receivers}
            base = {s: int(math.floor(ideals[s])) for s in receivers}
            rems = sorted(receivers, key=lambda t: ideals[t] - base[t], reverse=True)
            used = sum(base.values())
            for s in receivers:
                budgets[s] += base[s]
            for s in rems[: give - used]:
                budgets[s] += 1
        if debug:
            dist = ", ".join(f"{s}:{budgets[s]}" for s in series_order)
            print(f"[rebalance] new budgets -> {dist}")
        return True

    # Main growth loop
    iter_no = 0
    while any(budgets[s] > 0 for s in series_order):
        iter_no += 1
        progressed = False
        if debug:
            bud_str = ", ".join(f"{s}:{budgets[s]}" for s in series_order)
            print(f"[grow] iter={iter_no} budgets: {bud_str}")

        order = series_order[:]
        rng.shuffle(order)

        # Strict round first (majority must win)
        for s in order:
            if budgets[s] <= 0:
                continue
            # Batched high-support claims
            claims = 0
            while budgets[s] > 0 and claims < MAX_CLAIMS_HI:
                nb, sup, score = pick_best(s, min_sup=2, relaxed=False)
                if nb is None:
                    break
                assigned[nb] = s
                budgets[s] -= 1
                progressed = True
                claims += 1
                if debug:
                    cx, cy = centers[nb]
                    print(f"    -> {s} claims idx={nb} sup={sup} score={score:.3f} at ({float(cx):.3f}, {float(cy):.3f}) rem={budgets[s]}")
                # Update supports for this series
                for w in adj.get(nb, []):
                    if w not in comp_set or w in assigned:
                        continue
                    old = support_counts[s].get(w, 0)
                    new = old + 1
                    support_counts[s][w] = new
                    if 0 <= old <= 6:
                        candidate_buckets[s][old].discard(w)
                    if 0 <= new <= 6:
                        candidate_buckets[s][new].add(w)

            # Try at most one low-support claim (must still meet majority/tie)
            if budgets[s] > 0:
                nb, sup, score = pick_best(s, min_sup=0, relaxed=False)
                if nb is not None and sup is not None and sup <= 1:
                    assigned[nb] = s
                    budgets[s] -= 1
                    progressed = True
                    if debug:
                        cx, cy = centers[nb]
                        print(f"    -> {s} low-support idx={nb} sup={sup} score={score:.3f} at ({float(cx):.3f}, {float(cy):.3f}) rem={budgets[s]}")
                    for w in adj.get(nb, []):
                        if w not in comp_set or w in assigned:
                            continue
                        old = support_counts[s].get(w, 0)
                        new = old + 1
                        support_counts[s][w] = new
                        if 0 <= old <= 6:
                            candidate_buckets[s][old].discard(w)
                        if 0 <= new <= 6:
                            candidate_buckets[s][new].add(w)

        # If nothing happened, run a relaxed round to break stalemates (still avoids blatant minority grabs)
        if not progressed and any(budgets[s] > 0 for s in series_order):
            for s in order:
                if budgets[s] <= 0:
                    continue
                nb, sup, score = pick_best(s, min_sup=2, relaxed=True)
                if nb is None:
                    continue
                assigned[nb] = s
                budgets[s] -= 1
                progressed = True
                if debug:
                    cx, cy = centers[nb]
                    print(f"    -> {s} relaxed idx={nb} sup={sup} score={score:.3f} at ({float(cx):.3f}, {float(cy):.3f}) rem={budgets[s]}")
                for w in adj.get(nb, []):
                    if w not in comp_set or w in assigned:
                        continue
                    old = support_counts[s].get(w, 0)
                    new = old + 1
                    support_counts[s][w] = new
                    if 0 <= old <= 6:
                        candidate_buckets[s][old].discard(w)
                    if 0 <= new <= 6:
                        candidate_buckets[s][new].add(w)

        # If still no progress, try budget redistribution from stuck to frontier-rich series
        if not progressed and any(budgets[s] > 0 for s in series_order):
            changed = redistribute_leftovers()
            if not changed:
                # Nothing to do; break to completion step
                break

    # Connectivity-preserving completion:
    # Fill any remaining tiles by assigning to the series with max SUPPORT for that tile,
    # borrowing budget from stuck series if needed (keeps per-component totals exact).
    remaining = [i for i in comp_indices if i not in assigned]
    if remaining and debug:
        print(f"[finalize] remaining tiles to assign: {len(remaining)}")

    if remaining:
        # Precompute stuck list (those who still hold budget but have no frontier)
        def refresh_stuck():
            return [s for s in series_order if budgets[s] > 0 and series_frontier_size(s) == 0]

        for idx in remaining:
            # Choose series with maximum support for this tile
            best_sup, best_s = -1, None
            for s in series_order:
                sup = support_counts[s].get(idx, 0)
                if sup > best_sup:
                    best_sup, best_s = sup, s
            if best_s is None:
                # Fallback: pick any series with budget (should be rare)
                best_s = max(series_order, key=lambda s: budgets.get(s, 0))
            # Ensure we have budget: borrow from stuck if needed
            if budgets.get(best_s, 0) <= 0:
                donors = refresh_stuck()
                # If no stuck donors, borrow from the series with largest remaining budget
                if not donors:
                    donors = sorted(series_order, key=lambda s: budgets.get(s, 0), reverse=True)
                for d in donors:
                    if budgets.get(d, 0) > 0:
                        budgets[d] -= 1
                        budgets[best_s] = budgets.get(best_s, 0) + 1
                        if debug:
                            print(f"[finalize] borrow 1 from {d} -> {best_s}")
                        break
            assigned[idx] = best_s
            budgets[best_s] -= 1
            if debug:
                cx, cy = centers[idx]
                print(f"    -> finalize {best_s} idx={idx} sup={best_sup} at ({float(cx):.3f}, {float(cy):.3f}) rem={budgets[best_s]}")
            # Update supports for best_s neighbors (keeps later decisions cohesive)
            for w in adj.get(idx, []):
                if w not in comp_set or w in assigned:
                    continue
                old = support_counts[best_s].get(w, 0)
                new = old + 1
                support_counts[best_s][w] = new
                if 0 <= old <= 6:
                    candidate_buckets[best_s][old].discard(w)
                if 0 <= new <= 6:
                    candidate_buckets[best_s][new].add(w)

    return assigned


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
        raise RuntimeError(f"No tiles found for continent='{CONTINENT}'.")

    # Build a robust neighbor graph using file orientation
    adj = build_neighbor_graph(idx_continent, centers, radius, orientation, tol_factor=1e-3, debug=DEBUG)

    comps = connected_components(idx_continent.tolist(), adj)
    comps = [c for c in comps if c]
    comp_sizes = [len(c) for c in comps]

    total_tiles = int(sum(comp_sizes))
    global_counts = largest_remainder_apportion(DISTRIBUTION, total_tiles)
    per_comp_counts = split_counts_across_components(global_counts, comp_sizes)

    assigned_series: Dict[int, str] = {}
    series_names = list(DISTRIBUTION.keys())

    for ci, (comp_indices, local_counts) in enumerate(zip(comps, per_comp_counts)):
        if sum(local_counts.values()) == 0:
            continue
        if DEBUG:
            print(f"\n[comp {ci}] size={len(comp_indices)} local_counts={local_counts}")
        local_assigned = region_grow_component(
            comp_indices=comp_indices,
            adj=adj,
            series_names=series_names,
            local_counts=local_counts,
            centers=centers,
            rng=rng,
            debug=DEBUG,
        )
        assigned_series.update(local_assigned)

    # Ensure all continent tiles are assigned (fallback to largest series if not)
    unassigned = [i for i in idx_continent if i not in assigned_series]
    if unassigned:
        largest = max(global_counts.items(), key=lambda kv: kv[1])[0]
        for i in unassigned:
            assigned_series[i] = largest
        if DEBUG:
            print(f"[post] unassigned tiles: {len(unassigned)} -> assigned to '{largest}'")

    # Prepare subset centers and labels for HXF output
    centers_out = centers[idx_continent]
    series_per_idx = [assigned_series[int(i)] for i in idx_continent]
    class_is_land_out = np.ones(len(idx_continent), dtype=bool)  # all land in this subset

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
    main()