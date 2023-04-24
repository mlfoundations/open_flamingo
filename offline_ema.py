"""
Creates an EMA model from all checkpoints in a directory, and saves the EMA wrapper as a new checkpoint.

Format:
python offline_ema.py path_to_dir_containing_checkpoints --ema_beta 0.9999
"""

import argparse
import glob
from ema_pytorch import EMA
from open_flamingo import create_model_and_transforms
import torch
from open_flamingo.train.train_utils import get_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Path to directory containing checkpoints")
parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)
parser.add_argument(
    "--tokenizer_path",
    default="facebook/opt-30b",
    type=str,
    help="path to tokenizer",
)
parser.add_argument("--ema_beta", default=0.5, type=float)
parser.add_argument("--ema_power", default=1, type=float, help="Affects the growth rate of the EMA weights.")
parser.add_argument("--last_ckpt", default=None, type=int)
args = parser.parse_args()

# Get all checkpoints in directory
checkpoints = glob.glob(f"{args.path}/checkpoint_*.pt")
checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))

# Create EMA model from checkpoints
model, _, _ = create_model_and_transforms(
    args.vision_encoder_path,
    args.vision_encoder_pretrained,
    args.lm_path,
    args.tokenizer_path if args.tokenizer_path else args.lm_path,
)
ema_model = EMA(
    model, beta=args.ema_beta, update_every=1, update_after_step=0, include_online_model=False, power=args.ema_power, inv_gamma=1,
)

def load_checkpoint(path, model):
    ckpt = torch.load(path, map_location="cpu")
    ckpt = ckpt['model_state_dict']
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    # print(model.perceiver.latents)
    # print(ckpt['perceiver.latents'])


for checkpoint in checkpoints:
    ckpt_num = int(checkpoint.split("_")[-1].split(".")[0])
    if args.last_ckpt is not None and ckpt_num > args.last_ckpt:
        break
    print(f"{ckpt_num}\t{ema_model.get_current_decay()}\t{checkpoint}")
    load_checkpoint(checkpoint, model)
    ema_model.update()
    # print(model.lang_encoder.model.decoder.layers[9].gated_cross_attn_layer.attn_gate)
    # print(ema_model.ema_model.lang_encoder.model.decoder.layers[9].gated_cross_attn_layer.attn_gate)

# Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
ema_model.ema_model.requires_grad_(False)
ema_model.ema_model.perceiver.requires_grad_(True)
ema_model.ema_model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
ema_model.ema_model.lang_encoder.get_input_embeddings().requires_grad_(True)

# Save EMA model
torch.save({
    "model_state_dict": get_checkpoint(ema_model.ema_model),
    "step": ema_model.step,
}, f"/fsx/home-irena/opt1.3b-ema/ema_{args.ema_beta}_{'final' if args.last_ckpt is None else args.last_ckpt}.pt")

print("DONE")
