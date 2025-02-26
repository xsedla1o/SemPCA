import os
import random
import time

import numpy as np
import torch

print("Seeding everything...")
seed = 6
random.seed(seed)  # Python random module.
np.random.seed(seed)  # Numpy module.
torch.manual_seed(seed)  # Torch CPU random seed module.
torch.cuda.manual_seed(seed)  # Torch GPU random seed module.
torch.cuda.manual_seed_all(seed)  # Torch multi-GPU random seed module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONHASHSEED"] = str(seed)

print("Seeding Finished")

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SESSION = time.strftime("%y%m%d_%H%M%S", time.gmtime(time.time()))


def get_project_root():
    # The constants file is located in the root of the project
    package_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(package_root)


def get_logs_root():
    log_file_root = os.path.join(get_project_root(), "logs")
    return log_file_root


PROJECT_ROOT = get_project_root()

LOG_ROOT = os.environ.get("SEMPCA_LOG_DIR", get_logs_root())
os.makedirs(LOG_ROOT, exist_ok=True)
