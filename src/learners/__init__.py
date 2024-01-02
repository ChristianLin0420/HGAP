from .q_learner import QLearner
from .iqn_learner import IQNLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .diffusion_learner import DiffusionLearner
from .rest_q_learner_central import RestQLearnerCentral
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .nq_learner import NQLearner



REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["iqn_learner"] = IQNLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["diffusion_learner"] = DiffusionLearner
REGISTRY["restq_learner_central"] = RestQLearnerCentral
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["nq_learner"] = NQLearner

