from rep_algos.shared_training import learn_eigenvectors
from rep_algos import allo_learner
from src.config.ded_clf import Args
import tyro

args = tyro.cli(Args)
learn_eigenvectors(args, allo_learner)
