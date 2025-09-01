import math
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

from hxf_io import write_hxf
import numpy as np

import requests
import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_HEX_RADIUS_KM = 250.0 # Change this (smaller = more hexes, larger = fewer hexes)

WGS84_EPSG = 4326
EQUAL_EARTH_EPSG = 8857

NE_LAND_URL = "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip"
NE_ADMIN0_URL = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"



# -----------------------------
# Data download and loading
# -----------------------------
def download_and_extract(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    shp_files = list(out_dir.glob("*.shp"))
    if shp_files:
        return shp_files[0]

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        zf.extractall(out_dir)
    shp_files = list(out_dir.glob("*.shp"))
    if not shp_files:
        raise RuntimeError(f"Failed to find shapefile in {out_dir}")
    return shp_files[0]


def load_land_and_countries(data_root: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    land_shp = download_and_extract(NE_LAND_URL, data_root / "ne_110m_land")
    admin_shp = download_and_extract(NE_ADMIN0_URL, data_root / "ne_110m_admin_0_countries")

    land = gpd.read_file(land_shp).to_crs(epsg=WGS84_EPSG)
    countries = gpd.read_file(admin_shp).to_crs(epsg=WGS84_EPSG)

    land = land[~land.geometry.is_empty & land.geometry.is_valid].reset_index(drop=True)
    countries = countries[~countries.geometry.is_empty & countries.geometry.is_valid].reset_index(drop=True)
    return land, countries


# -----------------------------
# Hexagon math (pointy-top)
# -----------------------------
def hexagon_pointy(cx: float, cy: float, r: float) -> Polygon:
    pts = []
    for k in range(6):
        ang = math.radians(60 * k + 30)
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return Polygon(pts)


def make_hex_grid_pointy(bounds: tuple[float, float, float, float], hex_radius_m: float) -> gpd.GeoDataFrame:
    """
    Create a pointy-top hex grid covering bounds (Equal Earth meters).
    width = sqrt(3) * r, height = 2r, dx = width, dy = 1.5r, odd rows offset by width/2.
    """
    minx, miny, maxx, maxy = bounds
    r = hex_radius_m
    width = math.sqrt(3) * r
    height = 2.0 * r
    dx = width
    dy = 1.5 * r

    minx_pad, miny_pad = minx - width, miny - height
    maxx_pad, maxy_pad = maxx + width, maxy + height

    n_rows = int(math.ceil((maxy_pad - miny_pad) / dy)) + 1
    n_cols = int(math.ceil((maxx_pad - minx_pad) / dx)) + 1

    hexes = []
    for row in range(n_rows):
        cy = miny_pad + row * dy
        x_offset = (width / 2.0) if (row % 2 == 1) else 0.0
        for col in range(n_cols):
            cx = minx_pad + col * dx + x_offset
            poly = hexagon_pointy(cx, cy, r)
            if poly.intersects(box(minx, miny, maxx, maxy)):
                hexes.append(poly)

    return gpd.GeoDataFrame({"geometry": hexes}, crs=f"EPSG:{EQUAL_EARTH_EPSG}")


# -----------------------------
# Classification and metadata hooks
# -----------------------------
def classify_hexes_land_sea(hexes_eq: gpd.GeoDataFrame, land_eq: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Mark hexes as 'land' if centroid is within the land union, else 'sea'. CRS: Equal Earth.
    """
    land_union = unary_union(land_eq.geometry)
    land_prepared = prep(land_union)
    centroids = hexes_eq.geometry.centroid
    is_land = [land_prepared.contains(c) for c in centroids]

    out = hexes_eq.copy()
    out["class"] = ["land" if v else "sea" for v in is_land]
    return out


def add_hex_metadata(
    hexes_eq: gpd.GeoDataFrame,
    countries_wgs84: Optional[gpd.GeoDataFrame] = None
) -> gpd.GeoDataFrame:
    """
    Add 'continent' to hexes using Natural Earth admin-0 countries (field 'CONTINENT').
    Assigns continent only to land hexes (by centroid-in-country). Sea hexes get 'sea'.
    """
    if countries_wgs84 is None or "CONTINENT" not in countries_wgs84.columns:
        return hexes_eq

    # Work in Equal Earth to match hexes
    countries_eq = countries_wgs84[["CONTINENT", "geometry"]].to_crs(epsg=EQUAL_EARTH_EPSG)

    result = hexes_eq.copy()
    result["continent"] = None

    # Only classify land hexes
    land_mask = result["class"] == "land"
    if land_mask.any():
        land_hexes = result.loc[land_mask].copy()

        # Centroids for join
        land_pts = gpd.GeoDataFrame(
            land_hexes.drop(columns="geometry"),
            geometry=land_hexes.geometry.centroid,
            crs=hexes_eq.crs,
        )

        # Centroid within country -> inherit 'CONTINENT'
        joined = gpd.sjoin(land_pts, countries_eq, how="left", predicate="within")[["CONTINENT"]]

        # sjoin preserves left index, so we can align on index
        result.loc[land_mask, "continent"] = joined["CONTINENT"].values

    # Sea stays 'sea'; any unmatched land centroids remain None if they fell in gaps
    result["continent"] = result["continent"].fillna("sea")
    return result


# -----------------------------
# Pipeline
# -----------------------------

def save_hxf(
    hexes_eq: gpd.GeoDataFrame,
    hex_radius_m: float,
    out_path: Path
) -> None:
    centers = np.vstack([
        hexes_eq.geometry.centroid.x.values.astype(np.float32),
        hexes_eq.geometry.centroid.y.values.astype(np.float32),
    ]).T

    class_is_land = None
    if "class" in hexes_eq.columns:
        class_is_land = hexes_eq["class"].astype(str).str.lower().eq("land").values

    continent_names = None
    if "continent" in hexes_eq.columns:
        continent_names = hexes_eq["continent"].astype(object).where(lambda s: s.notna(), None).tolist()

    write_hxf(
        out_path,
        centers,
        epsg=EQUAL_EARTH_EPSG,
        radius_m=float(hex_radius_m),
        orientation=1,  # pointy-top
        class_is_land=class_is_land,
        continent_names=continent_names
    )

def save_layers(
    hexes_eq: gpd.GeoDataFrame,
    land_eq: gpd.GeoDataFrame,
    out_root: Path,
    hex_radius_m: float
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    hexes_eq.to_file(out_root / "world_hexes_equal_earth.gpkg", layer="hexes", driver="GPKG")
    hexes_wgs84 = hexes_eq.to_crs(epsg=WGS84_EPSG)
    hexes_wgs84.to_file(out_root / "world_hexes_wgs84.gpkg", layer="hexes", driver="GPKG")
    land_eq.to_file(out_root / "land_equal_earth.gpkg", layer="land", driver="GPKG")

    # New: compact HXF
    save_hxf(hexes_eq, hex_radius_m, out_root / "world_hexes_equal_earth.hxf")

def run(
    hex_radius_km: float = DEFAULT_HEX_RADIUS_KM,
    data_dir: str = "data",
    out_dir: str = "outputs",
    add_meta: bool = False
):
    data_root = Path(data_dir)
    out_root = Path(out_dir)

    # Load data in WGS84
    land_wgs84, countries_wgs84 = load_land_and_countries(data_root)

    # Project to Equal Earth
    land_eq = land_wgs84.to_crs(epsg=EQUAL_EARTH_EPSG)
    world_bounds_eq = land_eq.total_bounds

    # Generate grid and classify
    hexes_eq = make_hex_grid_pointy(tuple(world_bounds_eq), hex_radius_km * 1000.0)
    hexes_eq = classify_hexes_land_sea(hexes_eq, land_eq)

    # Optional metadata enrichment
    if add_meta:
        hexes_eq = add_hex_metadata(hexes_eq, countries_wgs84)

    # Save layers
    save_layers(hexes_eq, land_eq, out_root, hex_radius_m=hex_radius_km * 1000.0)


if __name__ == "__main__":
    print("Generating world hex grid...")
    run(
        hex_radius_km=DEFAULT_HEX_RADIUS_KM,
        data_dir="data",
        out_dir="outputs",
        add_meta=True
    )