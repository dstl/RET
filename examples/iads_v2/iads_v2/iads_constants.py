"""Constants used in the IADS scenario."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ret.space.culture import Culture
from ret.space.heightband import AbsoluteHeightBand

if TYPE_CHECKING:
    from ret.space.heightband import HeightBand

IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "./images"))

pos_coningsby = (30000, 300000, 0)
pos_amsterdam_north = (380000, 195000, 0)
pos_amsterdam_east = (385000, 190000, 0)
pos_amsterdam_south = (380000, 185000, 0)
pos_amsterdam_west = (375000, 190000, 0)
pos_amsterdam_center = (380000, 190000, 0)
pos_amsterdam_sensor_north = (380000, 200000, 0)
pos_amsterdam_sensor_west = (370000, 185000, 0)
pos_amsterdam_sensor_south = (380000, 180000, 0)

pos_amsterdam_center_500m = (380000, 190000, "500m")

height_bands: list[HeightBand] = [
    AbsoluteHeightBand("500m", 500),
    AbsoluteHeightBand("2000m", 2000),
    AbsoluteHeightBand("12000m", 12000),
]

culture_land = Culture("land")
culture_sea = Culture("sea")

pos_tolerance = 1000

patriot_icon_path = os.path.join(IMAGE_DIR, "patriot", "10061500001111090000.svg")
nasams_icon_path = os.path.join(IMAGE_DIR, "nasams", "10061500001111050000.svg")
radar_icon_path = os.path.join(IMAGE_DIR, "radar", "10061500002203000000.svg")
