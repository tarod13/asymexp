from typing import Annotated, Union

import tyro

from src.config.clf import ClfArgs
from src.config.allo import AlloArgs
from src.config.allo_ext import AlloExtArgs
from src.learners.shared_training import learn_eigenvectors
from src.learners import clf_learner_multi_mode, allo_learner, allo_ext_learner

LEARNERS = {
    ClfArgs:    (clf_learner_multi_mode, "clf"),
    AlloArgs:   (allo_learner,           "allo"),
    AlloExtArgs:(allo_ext_learner,       "allo"),
}

args = tyro.cli(Union[
    Annotated[ClfArgs,     tyro.conf.subcommand("clf")],
    Annotated[AlloArgs,    tyro.conf.subcommand("allo")],
    Annotated[AlloExtArgs, tyro.conf.subcommand("allo_ext")],
])
learner_module, method = LEARNERS[type(args)]
learn_eigenvectors(args, learner_module, method)
