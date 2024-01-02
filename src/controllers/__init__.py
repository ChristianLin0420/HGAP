REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_central_controller import CentralBasicMAC
from .hpn_controller import HPNMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["hpn_mac"] = HPNMAC
