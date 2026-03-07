from typing import Annotated, Union

import tyro

from src.config.clf import ClfArgs
from src.config.allo import AlloArgs
from rep_algos.shared_training import learn_eigenvectors
from rep_algos import clf_learner_multi_mode, allo_learner

LEARNERS = {
    ClfArgs: (clf_learner_multi_mode, "clf"),
    AlloArgs: (allo_learner, "allo"),
}

args = tyro.cli(Union[
    Annotated[ClfArgs, tyro.conf.subcommand("clf")],
    Annotated[AlloArgs, tyro.conf.subcommand("allo")],
])
learner_module, method = LEARNERS[type(args)]
learn_eigenvectors(args, learner_module, method)
