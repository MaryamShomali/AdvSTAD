"""Import the legacy CLI parser without consuming pytest's command line."""
import sys

import torch


_argv = sys.argv
try:
    sys.argv = [sys.argv[0]]
    import src.parser  # noqa: F401
finally:
    sys.argv = _argv

# Tiny attention fixtures are faster and reproducible without CPU thread fanout.
torch.set_num_threads(1)
