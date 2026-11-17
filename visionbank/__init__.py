"""VisionBank: an image-computable model of the human retina -> LGN -> V1 pathway, with voxel read-out and fitting."""
from .params import Params
from .model import Model, Run, UNITS, CHANNELS
from .frontend import Grid, CELLS
