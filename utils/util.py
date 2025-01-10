#Reference: https://github.com/RoyChao19477/SEMamba/blob/main/utils/util.py
import yaml
import torch
import os
import shutil
import glob
from torch.distributed import init_process_group

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def initialize_seed(seed):
    """Initialize the random seed for both CPU and GPU."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def print_gpu_info(num_gpus, cfg):
    """Print information about available GPUs and batch size per GPU."""
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
        print('Batch size per GPU:', int(cfg['training_cfg']['batch_size'] / num_gpus))

def initialize_process_group(cfg, rank):
    """Initialize the process group for distributed training."""
    #init_process_group(
    #    backend=cfg['env_setting']['dist_cfg']['dist_backend'],
    #    init_method=cfg['env_setting']['dist_cfg']['dist_url'],
    #    world_size=cfg['env_setting']['dist_cfg']['world_size'] * cfg['env_setting']['num_gpus'],
    #    rank=rank
    #)
    args.rank = int(os.environ['RANK'])
    args.gpu = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    os.environ['RANK'] = str(args.rank)
    os.environ['LOCAL_RANK'] = str(args.gpu)
    os.environ['WORLD_SIZE'] = str(args.world_size)

    init_process_group(
        backend=cfg['env_setting']['dist_cfg']['dist_backend'],
        init_method=cfg['env_setting']['dist_cfg']['dist_url'],
        world_size=int(os.environ['WORLD_SIZE']),
        rank=rank
    )

def log_model_info(rank, model, exp_path):
    """Log model information and create necessary directories."""
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print("Generator Parameters :", num_params)
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'logs'), exist_ok=True)
    print("checkpoints directory :", exp_path)

def load_ckpts(args, device):
    """Load checkpoints if available."""
    if os.path.isdir(args.exp_path):
        cp_g = scan_checkpoint(args.exp_path, 'g_')
        cp_do = scan_checkpoint(args.exp_path, 'do_')
        if cp_g is None or cp_do is None:
            return None, None, 0, -1
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        return state_dict_g, state_dict_do, state_dict_do['steps'] + 1, state_dict_do['epoch']
    return None, None, 0, -1

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????' + '.pth')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def build_env(config, config_name, exp_path):
    os.makedirs(exp_path, exist_ok=True)
    t_path = os.path.join(exp_path, config_name)
    if config != t_path:
        shutil.copyfile(config, t_path)

def load_optimizer_states(optimizers, state_dict_do):
    """Load optimizer states from checkpoint."""
    if state_dict_do is not None:
        optim_g, optim_d = optimizers
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])


import os
import torch
import numpy as np
import scipy.stats
from scipy.signal import butter, sosfilt

from pesq import pesq
from pystoi import stoi


def si_sdr_components(s_hat, s, n):
    # s_target
    alpha_s = np.dot(s_hat, s) / np.linalg.norm(s)**2
    s_target = alpha_s * s

    # e_noise
    alpha_n = np.dot(s_hat, n) / np.linalg.norm(n)**2
    e_noise = alpha_n * n

    # e_art
    e_art = s_hat - s_target - e_noise
    
    return s_target, e_noise, e_art

def energy_ratios(s_hat, s, n):
    s_target, e_noise, e_art = si_sdr_components(s_hat, s, n)

    si_sdr = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise + e_art)**2)
    si_sir = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise)**2)
    si_sar = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_art)**2)

    return si_sdr, si_sir, si_sar

def mean_conf_int(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

class Method():
    def __init__(self, name, base_dir, metrics):
        self.name = name
        self.base_dir = base_dir
        self.metrics = {} 
        
        for i in range(len(metrics)):
            metric = metrics[i]
            value = []
            self.metrics[metric] = value 
            
    def append(self, matric, value):
        self.metrics[matric].append(value)

    def get_mean_ci(self, metric):
        return mean_conf_int(np.array(self.metrics[metric]))

def hp_filter(signal, cut_off=80, order=10, sr=16000):
    factor = cut_off /sr * 2
    sos = butter(order, factor, 'hp', output='sos')
    filtered = sosfilt(sos, signal)
    return filtered

def si_sdr(s, s_hat):
    alpha = np.dot(s_hat, s)/np.linalg.norm(s)**2   
    sdr = 10*np.log10(np.linalg.norm(alpha*s)**2/np.linalg.norm(
        alpha*s - s_hat)**2)
    return sdr

def snr_dB(s,n):
    s_power = 1/len(s)*np.sum(s**2)
    n_power = 1/len(n)*np.sum(n**2)
    snr_dB = 10*np.log10(s_power/n_power)
    return snr_dB

def pad_spec(Y, mode="zero_pad"):
    T = Y.size(3)
    if T%64 !=0:
        num_pad = 64-T%64
    else:
        num_pad = 0
    if mode == "zero_pad":
        pad2d = torch.nn.ZeroPad2d((0, num_pad, 0,0))
    elif mode == "reflection":
        pad2d = torch.nn.ReflectionPad2d((0, num_pad, 0,0))
    elif mode == "replication":
        pad2d = torch.nn.ReplicationPad2d((0, num_pad, 0,0))
    else:
        raise NotImplementedError("This function hasn't been implemented yet.")
    return pad2d(Y)

def ensure_dir(file_path):
    directory = file_path
    if not os.path.exists(directory):
        os.makedirs(directory)


def print_metrics(x, y, x_hat_list, labels, sr=16000):
    _si_sdr_mix = si_sdr(x, y)
    _pesq_mix = pesq(sr, x, y, 'wb')
    _estoi_mix = stoi(x, y, sr, extended=True)
    print(f'Mixture:  PESQ: {_pesq_mix:.2f}, ESTOI: {_estoi_mix:.2f}, SI-SDR: {_si_sdr_mix:.2f}')
    for i, x_hat in enumerate(x_hat_list):
        _si_sdr = si_sdr(x, x_hat)
        _pesq = pesq(sr, x, x_hat, 'wb')
        _estoi = stoi(x, x_hat, sr, extended=True)
        print(f'{labels[i]}: {_pesq:.2f}, ESTOI: {_estoi:.2f}, SI-SDR: {_si_sdr:.2f}')

def mean_std(data):
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

def print_mean_std(data, decimal=2):
    data = np.array(data)
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    if decimal == 2:
        string = f'{mean:.2f} ± {std:.2f}'
    elif decimal == 1:
        string = f'{mean:.1f} ± {std:.1f}'
    return string

def set_torch_cuda_arch_list():
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPUs found.")
        return
    
    num_gpus = torch.cuda.device_count()
    compute_capabilities = []

    for i in range(num_gpus):
        cc_major, cc_minor = torch.cuda.get_device_capability(i)
        cc = f"{cc_major}.{cc_minor}"
        compute_capabilities.append(cc)
    
    cc_string = ";".join(compute_capabilities)
    os.environ['TORCH_CUDA_ARCH_LIST'] = cc_string
    print(f"Set TORCH_CUDA_ARCH_LIST to: {cc_string}")
