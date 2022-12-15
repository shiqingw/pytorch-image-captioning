import json
import platform
import torch
import argparse
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

from trainer import evaluate, generate_samples
import time

from dataloader import Flickr8KDataset
from decoder import CaptionDecoder
from utils_local.other_utils import set_up_causal_mask, plot_bleu_scores, plot_loss, format_time, save_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ResNet+Transformer')
    parser.add_argument('--exp_num', default=0, type=int, help='test case number')
    args = parser.parse_args()
    exp_num = args.exp_num
    result_dir = './results/exp_{:03d}'.format(exp_num)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    # Load the pipeline configuration file
    config_path = "./test_cases/config_{:03d}.json".format(exp_num)
    with open(config_path, "r", encoding="utf8") as f:
        config = json.load(f)

    if platform.system() == 'Darwin':
        if not torch.backends.mps.is_available():
            device = 'cpu'
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
        else:
            device = 'mps'
    else:
        use_gpu = config["use_gpu"] and torch.cuda.is_available()
        device = torch.device("cuda" if use_gpu else "cpu")
    print('==> device: ', device)

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

   # Define dataloader hyper-parameters
    train_hyperparams = {
        "batch_size": config["batch_size"]["train"],
        "shuffle": True,
        "num_workers": 1,
        "drop_last": True
    }

    # Create dataloaders
    train_set = Flickr8KDataset(config, config["split_save"]["train"], training=True)
    valid_set = Flickr8KDataset(config, config["split_save"]["validation"], training=False)
    test_set = Flickr8KDataset(config, config["split_save"]["test"], training=False)
    train_loader = DataLoader(train_set, **train_hyperparams)
    valid_loader = DataLoader(valid_set, **train_hyperparams)
    test_loader = DataLoader(test_set, **train_hyperparams)

    #######################
    # Set up the encoder 
    #######################
    # Download pretrained CNN encoder
    encoder = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
    # Extract only the convolutional backbone of the model
    encoder = torch.nn.Sequential(*(list(encoder.children())[:-2]))
    encoder = encoder.to(device)
    # Freeze encoder layers
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    ######################
    # Set up the decoder
    ######################
    # Instantiate the decoder
    decoder = CaptionDecoder(config)
    decoder = decoder.to(device)

    if config["checkpoint"]["load"]:
        checkpoint_path = config["checkpoint"]["path"]
        decoder.load_state_dict(torch.load(checkpoint_path))
    decoder.train()

    # Set up causal mask for transformer decoder
    causal_mask = set_up_causal_mask(config["max_len"], device)

    # Load training configuration
    train_config = config["train_config"]

    # Prepare the model optimizer
    if train_config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            decoder.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["l2_penalty"]
        )
    else:
        raise ValueError("Optimizer not defined!")
    
    # learning rate scheduler
    if train_config["scheduler"] == "None":
        scheduler = None
    elif train_config["scheduler"] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config["num_of_epochs"])
    else:
        raise ValueError("Scheduler not defined!")

    # Loss function
    loss_fcn = nn.CrossEntropyLoss(label_smoothing=0.1)
    train_loss = []
    validation_loss = []
    test_loss = []
    best_bleu_4 = 0
    validation_bleu_scores = {"bleu-1": [],
                "bleu-2": [],
                "bleu-3": [],
                "bleu-4": []}
    test_bleu_scores = {"bleu-1": [],
                "bleu-2": [],
                "bleu-3": [],
                "bleu-4": []}

    def train(epoch, train_loss): 
        decoder.train()
        epoch_train_loss = 0
        epoch_start_time = time.time()
        for batch_idx, (x_img, x_words, y, tgt_padding_mask) in enumerate(iter(train_loader)):
            optimizer.zero_grad()
            
            # Move the used tensors to defined device
            x_img, x_words = x_img.to(device), x_words.to(device)
            y = y.to(device)
            tgt_padding_mask = tgt_padding_mask.to(device)

            # Extract image features
            with torch.no_grad():
                img_features = encoder(x_img)
                img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
                img_features = img_features.permute(0, 2, 1)
                img_features = img_features.detach()

            # Get the prediction of the decoder
            y_pred = decoder(x_words, img_features, tgt_padding_mask, causal_mask)
            tgt_padding_mask = torch.logical_not(tgt_padding_mask)
            y_pred = y_pred[tgt_padding_mask]

            y = y[tgt_padding_mask]

            # Calculate the loss
            loss = loss_fcn(y_pred, y.long())

            # Update model weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), train_config["gradient_clipping"])
            optimizer.step()
            epoch_train_loss += loss.item()
            
        epoch_end_time = time.time()
        print("Epoch: {:03d} | Loss: {:.3f} | Training time: {}".format(epoch,
         epoch_train_loss/(batch_idx+1), format_time(epoch_end_time - epoch_start_time)))
        train_loss += [epoch_train_loss/(batch_idx+1)]
    
    def validate(epoch, validation_loss): 
        encoder.eval()
        decoder.eval()
        epoch_validation_loss = 0
        epoch_start_time = time.time()
        for batch_idx, (x_img, x_words, y, tgt_padding_mask) in enumerate(iter(valid_loader)):
            x_img, x_words = x_img.to(device), x_words.to(device)
            y = y.to(device)
            tgt_padding_mask = tgt_padding_mask.to(device)

            # Extract image features
            with torch.no_grad():
                img_features = encoder(x_img)
                img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
                img_features = img_features.permute(0, 2, 1)
                img_features = img_features.detach()

            # Get the prediction of the decoder
            y_pred = decoder(x_words, img_features, tgt_padding_mask, causal_mask)
            tgt_padding_mask = torch.logical_not(tgt_padding_mask)
            y_pred = y_pred[tgt_padding_mask]

            y = y[tgt_padding_mask]

            # Calculate the loss
            loss = loss_fcn(y_pred, y.long())
            epoch_validation_loss += loss.item()
            
        epoch_end_time = time.time()
        print("Epoch: {:03d} | Loss: {:.3f} | Validation time: {}".format(epoch,
         epoch_validation_loss/(batch_idx+1), format_time(epoch_end_time - epoch_start_time)))
        validation_loss += [epoch_validation_loss/(batch_idx+1)]
    
    def test(epoch, validation_loss): 
        encoder.eval()
        decoder.eval()
        epoch_test_loss = 0
        epoch_start_time = time.time()
        for batch_idx, (x_img, x_words, y, tgt_padding_mask) in enumerate(iter(test_loader)):
            x_img, x_words = x_img.to(device), x_words.to(device)
            y = y.to(device)
            tgt_padding_mask = tgt_padding_mask.to(device)

            # Extract image features
            with torch.no_grad():
                img_features = encoder(x_img)
                img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
                img_features = img_features.permute(0, 2, 1)
                img_features = img_features.detach()

            # Get the prediction of the decoder
            y_pred = decoder(x_words, img_features, tgt_padding_mask, causal_mask)
            tgt_padding_mask = torch.logical_not(tgt_padding_mask)
            y_pred = y_pred[tgt_padding_mask]

            y = y[tgt_padding_mask]

            # Calculate the loss
            loss = loss_fcn(y_pred, y.long())
            epoch_test_loss += loss.item()
            
        epoch_end_time = time.time()
        print("Epoch: {:03d} | Loss: {:.3f} | Testing time: {}".format(epoch,
         epoch_test_loss/(batch_idx+1), format_time(epoch_end_time - epoch_start_time)))
        validation_loss += [epoch_test_loss/(batch_idx+1)]

    start_time = time.time()
    for epoch in range(train_config["num_of_epochs"]):
        print("==> Epoch: {:03d}".format(epoch))
        train(epoch, train_loss)
        validate(epoch, validation_loss)
        test(epoch, test_loss)
        if scheduler != None:
            scheduler.step()

        evaluate(epoch, valid_set, encoder, decoder, config, device, validation_bleu_scores)
        evaluate(epoch, test_set, encoder, decoder, config, device, test_bleu_scores)

        new_bleu_4 = test_bleu_scores["bleu-4"][-1]
        if new_bleu_4 > best_bleu_4 :
            best_bleu_4 = new_bleu_4
            model_state = {
                    'epoch': epoch,
                    'test_loss': test_loss[-1],
                    'bleu-1': test_bleu_scores["bleu-1"][-1],
                    'bleu-2': test_bleu_scores["bleu-2"][-1],
                    'bleu-3': test_bleu_scores["bleu-3"][-1],
                    'bleu-4': test_bleu_scores["bleu-4"][-1],
                    'encoder_state_dict':encoder.state_dict(),
                    'decoder_state_dict':decoder.state_dict()

                }
            print("==> Saving...")
            torch.save(model_state, os.path.join(result_dir, "transformer_model_state.pth"))

            print("==> Drawing samples...")
            encoder.eval()
            decoder.eval()

            generate_samples(valid_set, encoder, decoder, config, device, result_dir, 'valid')
            generate_samples(test_set, encoder, decoder, config, device, result_dir, 'test')
            generate_samples(train_set, encoder, decoder, config, device, result_dir, 'train')
  
        
    stop_time = time.time()
    print("Total Time: %s" % format_time(stop_time - start_time))

    print("==> Saving training/validation/testing loss...")
    training_info = {"training_loss": train_loss, "validation_loss": validation_loss,
     "testing_loss": test_loss, "validation_bleu_scores": validation_bleu_scores, "test_bleu_scores": test_bleu_scores}
    save_dict(training_info, os.path.join(result_dir, "training_info.npy"))

    print("==> Drawing loss and bleu scores...")
    loss_path = os.path.join(result_dir, "loss.png")
    plot_loss(train_loss, validation_loss, test_loss, loss_path)

    plot_bleu_scores(validation_bleu_scores, os.path.join(result_dir, "validation_bleu_scores.png"))
    plot_bleu_scores(test_bleu_scores, os.path.join(result_dir, "test_bleu_scores.png"))

    print("==> Process finished.")


