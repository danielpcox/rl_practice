
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

import torch
import logging
import sys

from torchtyping import patch_typeguard
patch_typeguard()

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'info').upper()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def initialize_logging():
    print(f'Initializing logging with level {LOG_LEVEL}')
    numeric_level = getattr(logging, LOG_LEVEL, None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % LOG_LEVEL)
    logging.basicConfig(format='%(asctime)s/%(levelname)s: %(message)s',
                        stream=sys.stdout,
                        filemode='a',
                        level=numeric_level,
                        datefmt='%Y-%m-%d %H:%M:%S%Z',
                        force=True)


useMPS = False
if useMPS and torch.backends.mps.is_available():
    device = torch.device("mps")  # mps for my M1 Mac
else:
    device = torch.device("cpu")
print(torch.__version__)
print(device)


