import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from nltk.translate.bleu_score import corpus_bleu

from dataloader import Flickr8KDataset
from decoder import CaptionDecoder
from utils_local.decoding_utils import greedy_decoding
from utils_local.other_utils import save_checkpoint, log_gradient_norm, set_up_causal_mask, plot_bleu_scores, plot_loss, format_time


import matplotlib.pyplot as plt
import pickle
from matplotlib import rcParams
from textwrap import wrap

def save_image(img, full_path, caption=None):
    """Imshow for Tensor."""
    fig = plt.figure(figsize=(10, 10), dpi=100, frameon=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')
    ax = plt.subplot(111)

    if caption is not None:
        caption = caption.capitalize()
        caption = caption.replace("<unk>","[UNK]")
        title = ax.set_title("\n".join(wrap(caption, 45)), fontsize=30)

    #unnormalize 
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224 
    img[2] = img[2] * 0.225 
    img[0] += 0.485 
    img[1] += 0.456 
    img[2] += 0.406
    
    img = img.numpy().transpose((1, 2, 0))
    
    ax.imshow(img)
    ax.axis('off')
    # fig.tight_layout()
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)
    plt.savefig(full_path, dpi = 100)
    plt.close(fig)
    return


def evaluate(epoch, subset, encoder, decoder, config, device, bleu_scores_dict):
    
    batch_size = config["batch_size"]["eval"]
    max_len = config["max_len"]
    bleu_w = config["bleu_weights"]

    # Mapping from vocab index to string representation
    idx2word = subset._idx2word
    # Ids for special tokens
    sos_id = subset._start_idx
    eos_id = subset._end_idx
    pad_id = subset._pad_idx

    references_total = []
    predictions_total = []
    epoch_start_time = time.time()
    encoder.eval()
    decoder.eval()
    for x_img, y_caption in subset.inference_batch(batch_size):
        x_img = x_img.to(device)

        # Extract image features
        img_features = encoder(x_img)
        img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
        img_features = img_features.permute(0, 2, 1)
        img_features = img_features.detach()

        # Get the caption prediction for each image in the mini-batch
        predictions = greedy_decoding(decoder, img_features, sos_id, eos_id, pad_id, idx2word, max_len, device)
        references_total += y_caption
        predictions_total += predictions
    epoch_end_time = time.time()
    # Evaluate BLEU score of the generated captions
    bleu_1 = corpus_bleu(references_total, predictions_total, weights=bleu_w["bleu-1"]) * 100
    bleu_2 = corpus_bleu(references_total, predictions_total, weights=bleu_w["bleu-2"]) * 100
    bleu_3 = corpus_bleu(references_total, predictions_total, weights=bleu_w["bleu-3"]) * 100
    bleu_4 = corpus_bleu(references_total, predictions_total, weights=bleu_w["bleu-4"]) * 100
    bleu_scores_dict["bleu-1"].append(bleu_1)
    bleu_scores_dict["bleu-2"].append(bleu_2)
    bleu_scores_dict["bleu-3"].append(bleu_3)
    bleu_scores_dict["bleu-4"].append(bleu_4)
    print("Epoch: {:03d} | Bleu scores: {:.3f} | {:.3f} | {:.3f} | {:.3f} | Evaluation time: {}".format(epoch,
        bleu_1, bleu_2, bleu_3, bleu_4, format_time(epoch_end_time - epoch_start_time)))
    return


def generate_samples(subset, encoder, decoder, config, device, result_dir, subset_name):
    batch_size = config["batch_size"]["eval"]
    max_len = config["max_len"]
    bleu_w = config["bleu_weights"]

    # Mapping from vocab index to string representation
    idx2word = subset._idx2word
    # Ids for special tokens
    sos_id = subset._start_idx
    eos_id = subset._end_idx
    pad_id = subset._pad_idx

    encoder.eval()
    decoder.eval()

    for x_img, _ in subset.inference_batch(batch_size):
        for i in range(10):
            image = x_img[5*i:5*i+1]
            img_features = encoder(image.to(device))
            img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
            img_features = img_features.permute(0, 2, 1)
            img_features = img_features.detach()
            predictions = greedy_decoding(decoder, img_features, sos_id, eos_id, pad_id, idx2word, max_len, device)
            caption = ' '.join(predictions[0])
            save_image(image[0], os.path.join(result_dir, "{}_{:03d}".format(subset_name, i)), caption=caption)
        break
    return 
