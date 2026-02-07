from rep_algos.shared_training import learn_eigenvectors
from rep_algos import clf_learner  # or clf_skip_conj_learner, duals_learner
from src.config.ded_clf import Args
import tyro

args = tyro.cli(Args)
learn_eigenvectors(args, clf_learner)