import json
import platform
import torch
import argparse
import os
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

from trainer import save_image
from utils_local.decoding_utils import greedy_decoding

from dataloader import Flickr8KDataset
from decoder import CaptionDecoder

if __name__ == '__main__':

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
    
    checkpoint = torch.load(os.path.join(result_dir, "transformer_model_state.pth"))
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder = decoder.to(device)
    decoder.eval()

    test_set = Flickr8KDataset(config, config["split_save"]["test"], training=False)
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    preprocessing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config["image_specs"]["image_size"]),
            transforms.ToTensor(),
            normalize,
        ])


    batch_size = config["batch_size"]["eval"]
    max_len = config["max_len"]

    # Mapping from vocab index to string representation
    idx2word = test_set._idx2word
    # Ids for special tokens
    sos_id = test_set._start_idx
    eos_id = test_set._end_idx
    pad_id = test_set._pad_idx

    sample_list_path = "./dataset/flickr8k/Flickr_8k.testImages.txt"
    with open(sample_list_path, "r") as f:
        data = f.read()
    name_list = data.split("\n")
    for i in range(50):
        # print("==> Drawing sample {:03d} ...".format(i))
        img_name = name_list[i]
        img_location = os.path.join("./dataset/flickr8k/Images",img_name)
        img = Image.open(img_location).convert("RGB")
        image = preprocessing(img).unsqueeze(0)
        img_features = encoder(image.to(device))
        img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
        img_features = img_features.permute(0, 2, 1)
        img_features = img_features.detach()
        predictions = greedy_decoding(decoder, img_features, sos_id, eos_id, pad_id, idx2word, max_len, device)
        caption = ' '.join(predictions[0])
        save_image(image[0], os.path.join(result_dir, "sample_{:03d}".format(i)), caption=caption)
