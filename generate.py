import argparse, os, sys, glob, random
from typing import Tuple
import torch
import numpy as np
import copy
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from PIL.PngImagePlugin import PngInfo
import time, math
from datetime import datetime
import codecs

timetrace = {"start":time.time()}
device = "cuda"
# additional image separation (pixels of padding), between grid items.
GRID_IMAGE_SEPARATION = 10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(type=str, nargs="?", default="a painting of a painter painting a painting", help="text prompt for generation. Multiple prompts can be specified, separated by '||'", dest="prompt")
    parser.add_argument("-ii", "--init_img", type=str, default=None, help="use img2img mode. path to the input image", dest="init_img")
    parser.add_argument("-st", "--strength", type=float, default=0.75, help="init image strength for noising/de-noising. 1.0 corresponds to full destruction of initial information", dest="strength")
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default="outputs/generated")
    parser.add_argument("--skip_save", action='store_true', help="do not save individual samples. For speed measurements.", dest="skip_save")
    parser.add_argument("-s","--ddim_steps", type=int, default=50, help="number of ddim sampling steps", dest="ddim_steps")
    parser.add_argument("--plms", action='store_true', help="use plms sampling", dest="plms")
    parser.add_argument("--fixed_code", action='store_true', help="if enabled, uses the same starting code across samples ", dest="fixed_code")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling", dest="ddim_eta")
    parser.add_argument("--n_iter", type=int, default=1, help="sample this often")
    parser.add_argument("-H","--H", type=int, default=512, help="image height, in pixel space", dest="H")
    parser.add_argument("-W","--W", type=int, default=512, help="image width, in pixel space", dest="W")
    parser.add_argument("-C","--C", type=int, default=4, help="latent channels", dest="C")
    parser.add_argument("-f","--f", type=int, default=8, help="downsampling factor", dest="f")
    parser.add_argument("-n", "--n_samples", type=int, default=4, help="how many samples to produce for each given prompt. A.k.a. batch size", dest="n_samples")
    parser.add_argument("--n_rows", type=int, default=None, help="rows in the grid (default will create a square grid)", dest="n_rows")
    parser.add_argument("-cs", "--scale", type=float, default=7.5, help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))", dest="scale")
    parser.add_argument("--from-file", type=str, help="if specified, load prompts from this file")
    parser.add_argument("--config", type=str, default="optimizedSD/v1-inference.yaml", help="path to config which constructs model", dest="config")
    parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt", help="path to checkpoint of model", dest="ckpt")
    parser.add_argument("-S","--seed", type=int, default=None, help="the seed, for reproducible sampling (None will select a random seed)", dest="seed")
    parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
    parser.add_argument("--small_batch", action='store_true', help="Reduce inference time when generate a smaller batch of images", dest="small_batch")
    return parser.parse_args()

# any properties of opt specified here will be stored in PNG metadata, in addition to the prompt text.
EXTRA_ARGS_TO_STORE = ["ddim_steps", "plms", "fixed_code", "ddim_eta", "H", "W", "C", "f", "scale", "seed", "init_img", "strength"]

def main():
    opt = parse_args()
    timetrace = {"start":time.time()}
    #outpath = opt.outdir
    OUTPUTS_DIR = opt.outdir
    INDIVIDUAL_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "individual")
    UNPUB_DIR = os.path.join(OUTPUTS_DIR, "unpub")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(INDIVIDUAL_OUTPUTS_DIR, exist_ok=True)
    os.makedirs(UNPUB_DIR, exist_ok=True)
    TIMESTAMP = datetime.today().strftime('%Y%m%d%H%M%S')

    # write back the actual (random) seed if None was passed.
    opt.seed = seed_everything(opt.seed)

    # truncate incorrect input dimensions to a multiple of 64
    opt.H = int(opt.H/64.0)*64
    opt.W = int(opt.W/64.0)*64
    print(f"Image dimensions: {(opt.W,opt.H)}")

    batch_size = opt.n_samples
    do_autocast = opt.precision == "autocast"
    model, modelCS, modelFS = load_and_configure_model(opt.ckpt, opt.config, opt.ddim_steps, do_autocast, opt.small_batch)

    if opt.init_img is not None:
        init_latent = load_and_preprocess_image(opt.init_img, opt.H, opt.W, batch_size, modelFS, do_autocast)
    else:
        init_latent = None

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    if not opt.from_file:
        prompt:str = opt.prompt
        assert prompt is not None
        # prompt is now always a list, even when only one is specified.
        prompt = [p.strip() for p in prompt.split("||")]
        # repeat prompt batch_size times. limit to first batch_size elements incase multiple prompts are specified.
        data = [(batch_size*prompt)[:batch_size]]
        #data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    # gather up all prompts, deduplicate through set.
    all_prompts_for_grid = list(set(flatten_sublists(data)))
    # store metadata and prompt string containing all prompts (if multiples present) for the grid image
    all_prompts_for_filename, all_prompts_for_metadata = cleanup_str(all_prompts_for_grid)
    all_prompts_metadata, all_prompts_metadata_argdict = to_metadata(all_prompts_for_metadata, opt)

    if init_latent is not None:
        # prepare strength for init image decode
        assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(opt.strength * opt.ddim_steps)
        print(f"target t_enc is {t_enc} steps")
    else:
        t_enc = 0

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():

        images = []
        for n in trange(opt.n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = None
                    if opt.scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    
                    c = modelCS.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    mem = torch.cuda.memory_allocated()/1e6
                    modelCS.to("cpu")
                    while(torch.cuda.memory_allocated()/1e6 >= mem):
                        time.sleep(1)

                    if init_latent is not None:
                        # img2img mode
                        # encode init image latent (scaled latent)
                        z_enc = model.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        # decode into samples
                        samples_ddim = model.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc)
                    else:
                        # text2img mode
                        samples_ddim = model.sample(S=opt.ddim_steps,
                                        conditioning=c,
                                        batch_size=opt.n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=opt.scale,
                                        unconditional_conditioning=uc,
                                        eta=opt.ddim_eta,
                                        x_T=start_code)

                    modelFS.to(device)
                    print("saving images")
                    for i in range(batch_size):
                        prompt_for_filename, prompt_for_metadata = cleanup_str(prompts[i])
                        metadata, metadata_argdict = to_metadata(prompt_for_metadata, opt)
                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    # for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        images.append(img)
                        if not opt.skip_save:
                            img.save(os.path.join(INDIVIDUAL_OUTPUTS_DIR, f"{TIMESTAMP}_{prompt_for_filename}_{i}.png"), pnginfo=metadata)


                    mem = torch.cuda.memory_allocated()/1e6
                    modelFS.to("cpu")
                    while(torch.cuda.memory_allocated()/1e6 >= mem):
                        time.sleep(1)

                    del samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated()/1e6)

            grid_img = image_autogrid(images, opt.n_rows)
            base_filename = f"{TIMESTAMP}_{all_prompts_for_filename}"
            grid_img.save(os.path.join(OUTPUTS_DIR, f'{base_filename}.png'), pnginfo=all_prompts_metadata)
            with codecs.open(os.path.join(UNPUB_DIR, f"{base_filename}.txt"), mode="w", encoding="cp1252", errors="ignore") as f:
                f.write(all_prompts_for_metadata.replace("\n","\\n"))
                f.write("\n")
                f.write(str(all_prompts_metadata_argdict).replace("\n","\n"))

    timetrace["end"] = time.time()
    time_taken = (timetrace["end"] - timetrace["start"])/60.0
    print(f"Your samples are ready in {time_taken:.2f} minutes and waiting for you in {OUTPUTS_DIR} and {INDIVIDUAL_OUTPUTS_DIR}")

def load_and_configure_model(ckpt, config_path, ddim_steps=50, autocast=True, small_batch=False):
    sd = load_model_from_config(f"{ckpt}")
    li = []
    lo = []
    for key, value in sd.items():
        sp = key.split('.')
        if(sp[0]) == 'model':
            if('input_blocks' in sp):
                li.append(key)
            elif('middle_block' in sp):
                li.append(key)
            elif('time_embed' in sp):
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd['model1.' + key[6:]] = sd.pop(key)
    for key in lo:
        sd['model2.' + key[6:]] = sd.pop(key)
    
    config = OmegaConf.load(f"{config_path}")
    config.modelUNet.params.ddim_steps = ddim_steps
    
    if small_batch:
        config.modelUNet.params.small_batch = True
        print("Using small_batch mode!")
    else:
        config.modelUNet.params.small_batch = False
    
    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()
    
    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    
    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    
    if autocast:
        model.half()
        modelCS.half()
    return model, modelCS, modelFS

def load_and_preprocess_image(image_path, H, W, batch_size, modelFS, autocast=True):
    init_image = load_img(image_path, H, W).to(device)
    if autocast:
        init_image = init_image.half()
        modelFS.half()
    modelFS.to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))  # move to latent space
    mem = torch.cuda.memory_allocated()/1e6
    modelFS.to("cpu")
    while(torch.cuda.memory_allocated()/1e6 >= mem):
        time.sleep(1)
    return init_latent

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

def load_img(path, h0, w0):
    image = Image.open(path).convert("RGB")
    w, h = image.size

    print(f"loaded input image of size ({w}, {h}) from {path}")   
    if(h0 is not None and w0 is not None):
        h, w = h0, w0
    
    w, h = map(lambda x: x - x % 32, (w0, h0))  # resize to integer multiple of 32

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample = Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

# function to create one image containing all input images in a grid.
# currently not intended for images of differing sizes.
def image_autogrid(imgs, fixed_rows=None) -> Image:
    if fixed_rows is not None:
        rows = fixed_rows
        cols = math.ceil(len(imgs)/fixed_rows)
    elif len(imgs) == 3:
        cols=3
        rows=1
    else:
        side_len = math.sqrt(len(imgs))
        # round up cols from square root, attempt to round down rows
        # if required to actually fit all images, both cols and rows are rounded up.
        cols = math.ceil(side_len)
        rows = math.floor(side_len)
        if (rows*cols) < len(imgs):
            rows = math.ceil(side_len)
    # get grid item size from first image
    w, h = imgs[0].size
    # add separation to size between images as 'padding'
    w += GRID_IMAGE_SEPARATION
    h += GRID_IMAGE_SEPARATION
    # remove one image separation size from the overall size (no added padding after the final row/col)
    grid = Image.new('RGB', size=(cols*w-GRID_IMAGE_SEPARATION, rows*h-GRID_IMAGE_SEPARATION))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# cleanup string for use in a filename, extract string from list/tuple
def cleanup_str(input):
    if isinstance(input, (list, Tuple)) and len(input) == 1:
        s = str(input[0])
    else:
        s = str(input)
    new_string = "".join([char if char.isalnum() else "_" for char in s])
    # limit to something reasonable
    while "__" in new_string:
        new_string = new_string.replace("__","_")
    return new_string[:256], s

def to_metadata(prompt, ARGS) -> PngInfo:
    argdict = vars(ARGS)
    # only keep args with a specified value from the list of possible args
    argdict = {key:val for key, val in argdict.items() if key in EXTRA_ARGS_TO_STORE} #and not val in [None, "", []]}

    if not argdict["plms"]:
        argdict.pop("plms")
    if not argdict["fixed_code"]:
        argdict.pop("fixed_code")
    if argdict["H"] == 512:
        argdict.pop("H")
    if argdict["W"] == 512:
        argdict.pop("W")
    argdict["steps"] = argdict.pop("ddim_steps")
    if argdict["C"] == 4:
        argdict.pop("C")
    if argdict["f"] == 8:
        argdict.pop("f")
    if argdict["init_img"] is not None:
        argdict["init_img"] = True
    else:
        argdict.pop("init_img")
        argdict.pop("strength")

    argdict["method"] = "StableDiffusion"
    
    metadata_string = prompt.replace("\n","\\n")
    if len(argdict) > 0:
        metadata_string += "\n" + str(argdict).replace("\n", "\\n")
    
    metadata = PngInfo()
    metadata.add_text("prompt", metadata_string)
    return metadata, argdict

# takes in any l, flattens any list/tuple substructures, returns one list containing all leaf elements
def flatten_sublists(l:any) -> list:
    if not isinstance(l, (list, Tuple)):
        return [l]
    else:
        concatenated = []
        for item in l:
            concatenated += flatten_sublists(item)
        return concatenated

if __name__ == '__main__':
    main()