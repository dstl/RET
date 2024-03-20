"""Constants for the Example model."""
import os
from math import inf

from ret.sensing.perceivedworld import Confidence
from ret.space.culture import Culture
from ret.space.feature import BoxFeature, LineFeature, SphereFeature
from ret.space.heightband import AbsoluteHeightBand

IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "./images"))

culture_open = Culture("open", height=0.0, reflectivity=0.174)

culture_water = Culture("water", height=0.0, reflectivity=0.322, temperature=10.0)

culture_forest = Culture(
    "forest", height=10.0, reflectivity=0.122, wavelength_attenuation_factors={(400.0, 1200.0): 1.5}
)

culture_urban = Culture(
    "urban", height=15.0, reflectivity=0.228, wavelength_attenuation_factors={(400.0, 1200.0): 1.7}
)

ad_agent_area_centre = (6000, 8200)
ad_agent_area_centre_3d = (6000, 8200, "200m")
ad_agent_area_uav_drop_down_pos = (5950, 7500, "50m")
ad_agent_area = SphereFeature(
    center=ad_agent_area_centre, radius=600, name="AD Unit General Location"
)

approach_dogleg_position = (11000, 5300)

urban_centre = (10180, 11414)
urban_area = SphereFeature(center=urban_centre, radius=1000, name="Urban Area")
urban_approach = BoxFeature(min_coord=(9180, 8800), max_coord=(11180, 11414), name="Urban Approach")

ambush_line = LineFeature(coord_1=(9640, 7000), coord_2=(20000, 7000), name="Ambush line")
# TODO: Build roads as shown on the underlying terrain image

ground_gradient_speed_modifiers = [
    ((-inf, -1.1), 0.8),
    ((-1.1, 1.1), 1),
    ((1.1, inf), 0.8),
]

ground_culture_movement_modifiers = {
    culture_open: 1.0,
    culture_water: 0.0,
    culture_forest: 0.7,
    culture_urban: 0.8,
}

pos_uav_start = (5900, 4100, 500)
pos_rockets_start = (200, 200)

pos_ai_coy = (500, 500)

pos_ai_pl_1 = (900, 900)
pos_ai_sect_1_1 = (900, 1100)
pos_ai_sect_1_2 = (1000, 1000)
pos_ai_sect_1_3 = (1100, 900)

pos_ai_pl_2 = (1400, 900)
pos_ai_sect_2_1 = (1400, 1100)
pos_ai_sect_2_2 = (1500, 1000)
pos_ai_sect_2_3 = (1600, 900)

pos_ai_pl_3 = (1900, 900)
pos_ai_sect_3_1 = (1900, 1100)
pos_ai_sect_3_2 = (2000, 1000)
pos_ai_sect_3_3 = (2100, 900)

ai_pl_positions = [pos_ai_pl_1, pos_ai_pl_2, pos_ai_pl_3]

ai_pl_1_positions = [pos_ai_sect_1_1, pos_ai_sect_1_2, pos_ai_sect_1_3]
ai_pl_2_positions = [pos_ai_sect_2_1, pos_ai_sect_2_2, pos_ai_sect_2_3]
ai_pl_3_positions = [pos_ai_sect_3_1, pos_ai_sect_3_2, pos_ai_sect_3_3]

pos_anti_armour_start = (9640, 7470)

uav_height_bands = [
    AbsoluteHeightBand("50m", 50.0),
    AbsoluteHeightBand("200m", 200),
    AbsoluteHeightBand("500m", 500),
]
uav_position_tolerance = 50

ground_unit_position_tolerance = 20

contrast_performance_curve = [(0.00001, 0.00001), (0.2, 0.4), (0.5, 1.0), (1.0, 2.0)]

temperature_performance_curve = [(0.00001, 0.00001), (4.0, 0.4), (10.0, 1.0), (20.0, 2.0)]

johnson_criteria = {
    Confidence.DETECT: 0.75,
    Confidence.RECOGNISE: 4.0,
    Confidence.IDENTIFY: 8.0,
}
