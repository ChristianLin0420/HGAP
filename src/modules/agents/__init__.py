REGISTRY = {}

from .rnn_agent import RNNAgent
from .iqn_rnn_agent import IQNRNNAgent
from .central_rnn_agent import CentralRNNAgent
from .diffusion_rnn_agent import DiffusionRNNAgent
from .hpns_rnn_agent import HPNS_RNNAgent
from .hpns_attention_agent import HPNS_AttentionAgent
from .hgap_agent import HGAP_Agent
from .updet_agent import UPDeT


REGISTRY["rnn"] = RNNAgent
REGISTRY["iqn_rnn"] = IQNRNNAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["diffusion_rnn"] = DiffusionRNNAgent
REGISTRY["hpns_rnn"] = HPNS_RNNAgent
REGISTRY["hpns_attention"] = HPNS_AttentionAgent
REGISTRY["hgap"] = HGAP_Agent
REGISTRY['updet'] = UPDeT