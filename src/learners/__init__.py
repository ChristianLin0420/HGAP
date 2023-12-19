from .q_learner import QLearner
from .iqn_learner import IQNLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .diffusion_learner import DiffusionLearner
from .rest_q_learner_central import RestQLearnerCentral

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["iqn_learner"] = IQNLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["diffusion_learner"] = DiffusionLearner
REGISTRY["restq_learner_central"] = RestQLearnerCentral