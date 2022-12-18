import torch
import os
exp_nums = [1,2,3,4,5,6,7,8,9,10]
for exp_num in exp_nums:
    result_dir = './results/exp_{:03d}'.format(exp_num)
    checkpoint = torch.load(os.path.join(result_dir, "transformer_model_state.pth"))
    bleu_1 = checkpoint['bleu-1']/100.0
    bleu_2 = checkpoint['bleu-2']/100.0
    bleu_3 = checkpoint['bleu-3']/100.0
    bleu_4 = checkpoint['bleu-4']/100.0

    print("Test case: {:03d} | Bleu scores: {:.3f} | {:.3f} | {:.3f} | {:.3f}".format(exp_num,
         bleu_1, bleu_2, bleu_3, bleu_4))