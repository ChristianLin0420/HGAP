REGISTRY = {}

from .rnn_agent import RNNAgent
from .iqn_rnn_agent import IQNRNNAgent
from .diffusion_rnn_agent import DiffusionRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["iqn_rnn"] = IQNRNNAgent
REGISTRY["diffusion_rnn"] = DiffusionRNNAgent