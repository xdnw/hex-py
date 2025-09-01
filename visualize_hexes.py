from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.patches import Patch
from hxf_io import read_hxf

def generate_distinct_colors(n: int, cmap_name: str = "tab20") -> list[str]:
    if n <= 0:
        return []
    cmap = colormaps.get_cmap(cmap_name).resampled(n)
    return [to_hex(cmap(i)) for i in range(n)]

def _hex_vertices_pointy(radius_m: float) -> np.ndarray:
    # Six unit vertices for pointy-top hex, centered at origin
    angles = np.deg2rad(np.arange(6) * 60 + 30.0)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32) * float(radius_m)

def visualize_hxf(
    hxf_path: Path,
    sea_color: str = "#377eb8",
    default_land_color: str = "#4daf4a",
    other_color: str = "#999999",
    figsize: tuple[int, int] = (12, 7)
) -> None:
    data = read_hxf(hxf_path)
    centers = data["centers"]  # (N,2) float32
    r = float(data["radius_m"])

    # Colors
    n = centers.shape[0]
    class_mask = data.get("class_is_land")
    cont_labels = data.get("continent_labels")
    cont_codes = data.get("continent_codes")

    # Choose colors per cell
    colors = np.full(n, other_color, dtype=object)
    if class_mask is not None:
        colors[~class_mask] = sea_color
        colors[class_mask] = default_land_color
    mapping = {}
    if cont_labels is not None and cont_codes is not None:
        # Build palette ignoring empty labels
        unique = [lbl for lbl in cont_labels if lbl and lbl.strip()]
        palette = generate_distinct_colors(max(1, len(unique)))
        # Map label -> color
        pid = 0
        for lbl in cont_labels:
            if not lbl or not lbl.strip():
                mapping[lbl] = other_color
            else:
                mapping[lbl] = palette[pid % len(palette)]
                pid += 1
        # Apply
        labels_per_cell = [cont_labels[int(c)] for c in cont_codes]
        colors = np.array([mapping[lbl] for lbl in labels_per_cell], dtype=object)
        # If we also know sea/land, enforce sea color
        if class_mask is not None:
            colors[~class_mask] = sea_color

    # Build polygons vectorized: (N,6,2)
    unit_hex = _hex_vertices_pointy(r)  # (6,2)
    polys = centers[:, None, :] + unit_hex[None, :, :]

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    coll = PolyCollection(polys, facecolors=colors, edgecolors="#333333", linewidths=0.1, alpha=0.95)
    ax.add_collection(coll)

    # Legend for continents (if available)
    if cont_labels is not None and cont_codes is not None:
        # Labels assigned per cell
        labels_per_cell = [cont_labels[int(c)] for c in cont_codes]

        from collections import Counter

        # Count occurrences per label, using only land tiles if class_mask exists
        if class_mask is not None:
            land_idxs = np.nonzero(class_mask)[0]
            land_labels_list = [labels_per_cell[i] for i in land_idxs if
                                labels_per_cell[i] and labels_per_cell[i].strip()]
            counts = Counter(land_labels_list)
            total_relevant = sum(counts.values())  # total land tiles with a continent label
        else:
            labeled_list = [lbl for lbl in labels_per_cell if lbl and lbl.strip()]
            counts = Counter(labeled_list)
            total_relevant = sum(counts.values())  # total labeled tiles

        # Visible continent labels in original order that have at least one counted tile
        visible_labels = [lbl for lbl in cont_labels if lbl and lbl.strip() and counts.get(lbl, 0) > 0]

        # Build legend entries: Sea (if available) then continents with counts and percents
        if visible_labels or (class_mask is not None and (~class_mask).any()):
            legend_patches = []

            # Continent entries: count and percent relative to total_relevant (land or labeled tiles)
            for lbl in visible_labels:
                cnt = counts.get(lbl, 0)
                # pct = 100.0 * cnt / total_relevant if total_relevant else 0.0
                legend_patches.append(
                    Patch(facecolor=mapping[lbl], edgecolor="#333333", label=f"{lbl} ({cnt} tiles)") #  {pct:.1f}%)
                )

            ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1.01, 1), title="Continents",
                      frameon=False)

    # Set bounds
    minx, miny = polys.reshape(-1, 2).min(axis=0)
    maxx, maxy = polys.reshape(-1, 2).max(axis=0)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Equal-area World Hex Grid (HXF)")
    plt.show()


if __name__ == "__main__":
    visualize_hxf(Path("outputs/south america_series_assignment.hxf"))