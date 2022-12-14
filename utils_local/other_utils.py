import os
import torch
import matplotlib.pyplot as plt
import pickle
from matplotlib import rcParams
from textwrap import wrap


def set_up_causal_mask(seq_len, device):
    """Defines the triangular mask used in transformers.

    This mask prevents decoder from attending the tokens after the current one.

    Arguments:
        seq_len (int): Maximum length of input sequence
        device: Device on which to map the created tensor mask
    Returns:
        mask (torch.Tensor): Created triangular mask
    """
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
    mask.requires_grad = False
    return mask


def log_gradient_norm(model, writer, step, mode, norm_type=2):
    """Writes model param's gradients norm to tensorboard"""
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    writer.add_scalar(f"Gradient/{mode}", total_norm, step)


def save_checkpoint(model, optimizer, start_time, epoch):
    """Saves specified model checkpoint."""
    target_dir = os.path.join("checkpoints", str(start_time))
    os.makedirs(target_dir, exist_ok=True)
    # Save model weights
    save_path_model = os.path.join(target_dir, f"model_{epoch}.pth")
    save_path_optimizer = os.path.join(target_dir, f"optimizer_{epoch}.pth")
    torch.save(model.state_dict(), save_path_model)
    torch.save(optimizer.state_dict(), save_path_optimizer)
    print("Model saved.")

def plot_loss(train_loss, validation_loss, test_loss, loss_path):
    fig = plt.figure(figsize=(10, 6), dpi=100, frameon=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.subplot(111)
    ax.set_xlabel("epoch", fontsize=50)
    ax.set_ylabel("loss", fontsize=50)
    ax.plot(train_loss, linewidth = 5, label="training loss")
    ax.plot(validation_loss, linewidth = 5, label="validation loss")
    ax.plot(test_loss, linewidth = 5, label="testing loss")
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.grid()
    ax.legend(fontsize = 40)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=100)
    plt.close(fig)

def plot_bleu_scores(bleu_dict, loss_path):
    fig = plt.figure(figsize=(10, 6), dpi=100, frameon=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.subplot(111)
    ax.set_xlabel("epoch", fontsize=50)
    ax.set_ylabel("bleu scores", fontsize=50)
    ax.plot(bleu_dict["bleu-1"], linewidth = 5, label="bleu-1")
    ax.plot(bleu_dict["bleu-2"], linewidth = 5, label="bleu-2")
    ax.plot(bleu_dict["bleu-3"], linewidth = 5, label="bleu-3")
    ax.plot(bleu_dict["bleu-4"], linewidth = 5, label="bleu-4")
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.grid()
    ax.legend(fontsize = 40)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=100)
    plt.close(fig)

def save_dict(dict_obj, fullname):
    with open(fullname, 'wb') as handle:
        pickle.dump(dict_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(fullname):
    with open(fullname, 'rb') as handle:
        loaded_obj = pickle.load(handle)
    return loaded_obj


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

