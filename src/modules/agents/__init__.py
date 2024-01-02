REGISTRY = {}

from .rnn_agent import RNNAgent
from .iqn_rnn_agent import IQNRNNAgent
from .central_rnn_agent import CentralRNNAgent
from .diffusion_rnn_agent import DiffusionRNNAgent
from .hpns_rnn_agent import HPNS_RNNAgent


REGISTRY["rnn"] = RNNAgent
REGISTRY["iqn_rnn"] = IQNRNNAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["diffusion_rnn"] = DiffusionRNNAgent
REGISTRY["hpns_rnn"] = HPNS_RNNAgent
