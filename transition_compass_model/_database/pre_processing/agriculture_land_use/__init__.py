"""Preprocessing for Agriculture Land Use."""

# Note: canton_agriculture_pre-processing cannot be imported due to hyphen in filename
# from . import canton_agriculture_pre-processing
from . import (
    agriculture_landuse_preprocessing_EU,
    agriculture_landuse_preprocessing_VD,
    agriculture_preprocessing_temp,
    api_FAO_example,
    forestry_preprocessing_ch,
)

__all__ = [
    "agriculture_landuse_preprocessing_EU",
    "agriculture_landuse_preprocessing_VD",
    "agriculture_preprocessing_temp",
    "api_FAO_example",
    # "canton_agriculture_pre-processing",  # Cannot import due to hyphen
    "forestry_preprocessing_ch",
]
