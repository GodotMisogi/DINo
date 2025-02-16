# %%
import matplotlib
from ode_model import Decoder, Derivative
from network import MLP, SetEncoder
from datetime import datetime
from torchdiffeq import odeint
from torchsummary import summary
import getopt
import sys
import os
import logging
from torch import nn
import matplotlib.pyplot as plt
import torch
from utils import (
    process_config,
    count_parameters,
    set_rdm_seed,
    create_logger,
    scheduling,
    write_image,
    eval_dino,
    eval_dino_cond,
)
from matplotlib import rcParams

logging.getLogger("numba").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("PIL").setLevel(logging.WARNING)
matplotlib.pyplot.set_loglevel("critical")

log_every = 8
input_dataset = "storm_surge"
gpu = 0
gpu_id = 0
home_folder = "./results"
lr = 1e-2
lr_adapt = 1e-2
seed = 1
options = {}
# opts, args = getopt.getopt(sys.argv[1:], "c:d:f:g:r:w:")
subsampling_rate = 1.0
checkpoint_path = None  # warm start from a model in this path
n_cond = 0
# for opt, arg in opts:
#     if opt == "-c":
#         checkpoint_path = arg
#     if opt == "-d":
#         input_dataset = arg
#     if opt == "-f":
#         home_folder = arg
#     if opt == "-g":
#         gpu = int(arg)
#     if opt == "-r":
#         subsampling_rate = float(arg)
#     if opt == "-w":
#         n_cond = int(arg)

mask_data = 1.0 - subsampling_rate
now = datetime.now()
ts = now.strftime("%Y%m%d_%H%M%S")

cuda = torch.cuda.is_available()
if cuda:
    gpu_id = gpu
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

path_results = os.path.join(home_folder, input_dataset)
path_checkpoint = os.path.join(path_results, ts)
logger = create_logger(path_checkpoint, os.path.join(path_checkpoint, "log"))
os.makedirs(path_checkpoint, exist_ok=True)
init_type = "default"
set_rdm_seed(seed)

# Config
first = 4
n_frames_train = 10
(
    mask,
    mask_ts,
    size,
    state_dim,
    coord_dim,
    code_dim,
    hidden_c,
    hidden_c_enc,
    n_layers,
    dataset_tr_params,
    dataset_tr_eval_params,
    dataset_ts_params,
    dataloader_tr,
    dataloader_tr_eval,
    dataloader_ts,
) = process_config(
    input_dataset,
    path_results,
    mask_data=mask_data,
    device=device,
    n_frames_train=n_frames_train,
)
epsilon = epsilon_t = 0.99
eval_every = 100
n_epochs = 120000
method = "rk4" if n_cond == 0 else "euler"

if input_dataset == "wave" or input_dataset == "shallow_water":
    n_steps = 500
else:
    n_steps = 300

if checkpoint_path is None:  # Start from scratch
    # Decoder
    net_dec_params = {
        "state_c": state_dim,
        "code_c": code_dim,
        "hidden_c": hidden_c_enc,
        "n_layers": n_layers,
        "coord_dim": coord_dim,
    }
    # Forecaster
    net_dyn_params = {
        "state_c": state_dim,
        "hidden_c": hidden_c,
        "code_c": code_dim if n_cond == 0 else code_dim * 2,
    }
    net_dec = Decoder(**net_dec_params)
    net_dyn = Derivative(**net_dyn_params)
    if n_cond > 0:
        net_cond_params = {
            "code_size": code_dim * state_dim,
            "n_cond": n_cond,
            "hidden_size": 1024,
        }
        net_cond = SetEncoder(**net_cond_params)
    states_params = nn.ParameterList(
        [
            nn.Parameter(torch.zeros(n_frames_train, code_dim * state_dim).to(device))
            for _ in range(dataset_tr_eval_params["n_seq"])
        ]
    )

    print(dict(net_dec.named_parameters()).keys())
    print(dict(net_dyn.named_parameters()).keys())

    net_dec = net_dec.to(device)
    net_dyn = net_dyn.to(device)
    if n_cond > 0:
        net_cond = net_cond.to(device)
else:  # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{gpu_id}")
    logger.info(f"N_ones: {torch.sum(mask_ts)}")
    logger.info(
        f"Missingness: {100. * (1 - torch.sum(mask_ts) / size)}%"
    )
    net_dec_params = checkpoint["net_dec_params"]
    state_dim = net_dec_params["state_c"]
    code_dim = net_dec_params["code_c"]
    net_dec = Decoder(**net_dec_params)
    net_dec_dict = net_dec.state_dict()
    pretrained_dict = {
        k: v for k, v in checkpoint["dec_state_dict"].items() if k in net_dec_dict
    }
    net_dec_dict.update(pretrained_dict)
    net_dec.load_state_dict(net_dec_dict)
    print(dict(net_dec.named_parameters()).keys())

    net_dyn_params = checkpoint["net_dyn_params"]
    net_dyn = Derivative(**net_dyn_params)
    net_dyn_dict = net_dyn.state_dict()
    pretrained_dict = {
        k: v for k, v in checkpoint["dyn_state_dict"].items() if k in net_dyn_dict
    }
    net_dyn_dict.update(pretrained_dict)
    net_dyn.load_state_dict(net_dyn_dict)
    print(dict(net_dyn.named_parameters()).keys())

    if n_cond > 0:
        net_cond = SetEncoder(code_dim, n_cond, 1024)
        net_cond_dict = net_cond.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint["cond_state_dict"].items() if k in net_dyn_dict
        }
        net_cond_dict.update(pretrained_dict)
        net_cond.load_state_dict(net_cond_dict)
        print(dict(net_cond.named_parameters()).keys())

    states_params = checkpoint["states_params"]
    net_dec = net_dec.to(device)
    net_dyn = net_dyn.to(device)
    if n_cond > 0:
        net_cond = net_cond.to(device)

criterion = nn.MSELoss()

optim_net_dec = torch.optim.Adam([{"params": net_dec.parameters(), "lr": lr}])
optim_net_dyn = torch.optim.Adam([{"params": net_dyn.parameters(), "lr": lr / 10}])
optim_states = torch.optim.Adam([{"params": states_params, "lr": lr / 10}])
if n_cond:
    optim_net_cond = torch.optim.Adam(
        [{"params": net_cond.parameters(), "lr": lr / 10}]
    )


# summary(net_dec, [(1, size[0], size[1], coord_dim)])
# summary(net_dyn, [(1, state_dim, code_dim)])

# Logs
logger.info(f"run_id: {ts}")
if cuda:
    logger.info(f"gpu_id: {gpu_id}")
logger.info(f"seed: {seed}")
logger.info(f"dataset: {input_dataset}")
logger.info(f"method: {method}")
logger.info(f"code_c: {code_dim}")
logger.info(f"lr: {lr}")
logger.info(
    f"n_params forecaster: {count_parameters(net_dec) + count_parameters(net_dyn)}"
)
logger.info(f"coord_dim: {coord_dim}")
logger.info(f"n_frames_train: {n_frames_train}")
logger.info(f"subsampling_rate: {subsampling_rate*100}%")
if n_cond > 0:
    logger.info(f"n_cond: {n_cond}")
# %%
