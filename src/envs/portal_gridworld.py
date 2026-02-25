"""
Backward-compatibility shim for portal environments.

Portal logic is now built directly into GridWorldEnv via the `portals`
constructor argument.  This module exists so that existing imports of
PortalGridWorldEnv continue to work.
"""

from src.envs.gridworld import GridWorldEnv

# Alias — PortalGridWorldEnv is just GridWorldEnv now.
PortalGridWorldEnv = GridWorldEnv
