from rep_algos.shared_training import learn_eigenvectors
from rep_algos import clf_learner_multi_mode, allo_learner
from src.config.ded_clf import Args
import tyro

LEARNERS = {
    "clf": clf_learner_multi_mode,
    "allo": allo_learner,
}

args = tyro.cli(Args)
if args.algo not in LEARNERS:
    raise ValueError(f"Unknown algo '{args.algo}'. Options: {list(LEARNERS)}")
learn_eigenvectors(args, LEARNERS[args.algo])
