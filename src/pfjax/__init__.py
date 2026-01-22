# from . __metadata__ import __version__, __author__
from importlib.metadata import PackageNotFoundError, version

from .loglik_full import loglik_full
from .models.base_model import BaseModel
from .particle_filter import particle_filter, particle_filter_rb
from .particle_smooth import particle_smooth
from .simulate import simulate

# __version__ = "0.0.3rc1"
try:
    __version__ = version("pfjax")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

__author__ = "Martin Lysy, Pranav Subramani, Jonathan Ramkissoon, Mohan Wu, Michelle Ko, Kanika Choptra, Feiyu Zhu, Micky Liu, Monica Zhu"

__all__ = [
    "BaseModel",
    "simulate",
    "loglik_full",
    "particle_filter",
    "particle_filter_rb",
    "particle_smooth",
]
