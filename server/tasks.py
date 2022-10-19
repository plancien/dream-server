from celery import Celery
from celery.signals import worker_process_init

from sys import path as sys_path


sys_path.append('../stable-diffusion')


# for CUDA multiprocessing issue, run celery as solo
# https://stackoverflow.com/questions/63645357/using-pytorch-with-celery

cel_app = Celery('tasks', broker='redis://localhost/1', backend='redis://localhost/2')


model  = None
device = None
config = None

plms_sampler = None
ddim_sampler = None

inpainting_config  = None
inpainting_model   = None
inpainting_sampler = None


import os
import torch
from numpy import uint8
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything


# removes init model warning with weights
from transformers import logging
logging.set_verbosity_error()


import utils
import toxicode_utils

from ddim_simplified import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


device = torch.device("cuda")

torch.set_default_tensor_type(torch.HalfTensor)


def init():
    global device
    global config
    global model
    config, model = utils.load_config_and_model(
        "../models/v1-inference.yaml",
        "../models/sd-v1-4.ckpt",
        device
    )

    global plms_sampler
    global ddim_sampler
    plms_sampler = PLMSSampler(model)
    ddim_sampler = DDIMSampler(model)


def init_inpainting():
    global inpainting_config
    global inpainting_model
    global inpainting_sampler
    inpainting_config, inpainting_model = utils.load_inpainting_config_and_model(
        "../models/inpainting_config.yaml",
        "../models/inpainting_last.ckpt",
        device
    )

    inpainting_sampler = DDIMSampler(inpainting_model) #FIXME try plms, https://github.com/Sanster/lama-cleaner/releases/tag/0.13.0


def choose_sampler (opt):
    if opt.plms:
        if opt.image_guide:
            raise NotImplementedError("PLMS sampler not (yet) supported")  # FIXME check ?
        return plms_sampler
    else:
        return ddim_sampler


def run_txt2img(model, opt):
    from types import SimpleNamespace
    opt = SimpleNamespace(**opt)

    print(f"txt2img seed: {opt.seed}   steps: {opt.steps}  prompt: {opt.prompt}")
    print(f"size:  {opt.W}x{opt.H}")


    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    # torch.manual_seed(seeds[accelerator.process_index].item())

    sampler = choose_sampler(opt)

    # model_wrap = K.external.CompVisDenoiser(model)
    # sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()


    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    base_count = len(os.listdir(sample_path))


    batch_size = opt.n_samples
    n_rows     = opt.n_rows if opt.n_rows > 0 else batch_size

    prompts_data = utils.get_prompts_data(opt)

    grid_path = ''


    image_guide  = None
    latent_guide = None

    t_start = None
    masked_image_for_blend  = None
    mask_for_reconstruction = None
    latent_mask_for_blend   = None

    # this explains the [1, 4, 64, 64]
    shape = (batch_size, opt.C, opt.H//opt.f, opt.W//opt.f)

    sampler.make_schedule(ddim_num_steps=opt.steps, ddim_eta=opt.ddim_eta, verbose=False)

    if opt.image_guide:
        image_guide  = utils.image_path_to_torch(opt.image_guide, device)  # [1, 3, 512, 512]
        latent_guide = utils.torch_image_to_latent(model, image_guide, n_samples=opt.n_samples)  # [1, 4, 64, 64]


    if opt.blend_mask:
        [mask_for_reconstruction, latent_mask_for_blend] = toxicode_utils.get_mask_for_latent_blending(
            device, opt.blend_mask, blur=opt.mask_blur)  # [512, 512]  [1, 4, 64, 64]
        if not opt.return_changes_only:
            masked_image_for_blend = (
                1 - mask_for_reconstruction) * image_guide[0]  # [3, 512, 512]

    elif image_guide is not None:
        assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_start = int(opt.strength * opt.steps)

        print(f"target t_start is {t_start} steps")


    multiple_mode = (opt.n_iter * len(prompts_data) * opt.n_samples > 1)

    with torch.no_grad(), model.ema_scope(), torch.cuda.amp.autocast():
        tic = time.time()
        all_samples = list()
        counter = 0
        for n in trange(opt.n_iter, desc="Sampling"):
            for prompts in tqdm(prompts_data, desc="data"):

                seed = opt.seed + counter
                seed_everything(seed)

                unconditional_conditioning, conditioning = utils.get_conditionings(model, prompts, opt)

                samples = sampler.ddim_sampling(
                    conditioning,               # [1, 77, 768]
                    shape,  # (1, 4, 64, 64)
                    x0=latent_guide,            # [1, 4, 64, 64]
                    mask=latent_mask_for_blend,  # [1, 4, 64, 64]
                    # 12 (if 20 steps and strength 0.75 => 15)
                    t_start=t_start,
                    unconditional_guidance_scale=opt.scale,
                    # [1, 77, 768]
                    unconditional_conditioning=unconditional_conditioning,
                )  # [1, 4, 64, 64]

                x_samples = utils.encoded_to_torch_image(
                    model, samples)  # [1, 3, 512, 512]

                if opt.blend_mask is not None:
                    if opt.return_changes_only:
                        x_samples = torch.cat((x_samples, mask_for_reconstruction.unsqueeze(0).unsqueeze(0)), dim=1)
                    else:
                        x_samples = mask_for_reconstruction * x_samples + masked_image_for_blend
                
                all_samples.append(x_samples)

                if (not opt.skip_save) and (not multiple_mode):
                    for x_sample in x_samples:
                        image = utils.sampleToImage(x_sample)

                        image.save(
                            os.path.join(sample_path, f"{base_count:05}.png"),
                            pnginfo=toxicode_utils.metadata(
                                opt,
                                prompt=prompts[0],  # FIXME [0]
                                seed=seed,
                                generation_time=generated_time - tic
                            ))

                        base_count += 1


        if not opt.skip_grid:

            generated_time = time.time()

            if multiple_mode:
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                #grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                #image = Image.fromarray(grid.astype(np.uint8))
            else:
                grid = all_samples[0][0]

            image = utils.sampleToImage(grid)
            grid_path = os.path.join(outpath, f'{opt.file_prefix}-0000.png')

            image.save(grid_path, pnginfo=toxicode_utils.metadata(
                opt,
                prompt = prompts[0],  # FIXME [0]
                seed   = opt.seed,
                generation_time = generated_time - tic
                ))

        toc = time.time()

        counter += 1

    #FIXME at the end ? at the beginning ?
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print(f"Sampling took {toc-tic:g}s, i.e. produced {opt.n_iter * opt.n_samples / (toc - tic):.2f} samples/sec.")

    return [image, grid_path]




def run_inpaint(model, image, steps):
    result = 0
    with torch.no_grad(), model.ema_scope(), torch.cuda.amp.autocast():
        batch = toxicode_utils.make_inpaint_batch(image, device=device)

        # encode masked image and concat downsampled mask
        conditioning = model.cond_stage_model.encode(batch["masked_image"])
        cc = torch.nn.functional.interpolate(batch["mask"], size=conditioning.shape[-2:])
        conditioning = torch.cat((conditioning, cc), dim=1)

        size = (conditioning.shape[0], conditioning.shape[1]-1,) + conditioning.shape[2:]

        inpainting_sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0., verbose=False)

        samples_ddim = inpainting_sampler.ddim_sampling(conditioning, size)

        predicted_image = utils.encoded_to_torch_image(model, samples_ddim)

        image = torch.clamp((batch["image"]+1.0)/2.0, min=0.0, max=1.0)
        mask  = torch.clamp((batch["mask"]+1.0)/2.0, min=0.0, max=1.0)

        inpainted = (1-mask)*image + mask*predicted_image

        #FIXME use util ?
        inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
        result = Image.fromarray(inpainted.astype(uint8))

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return result



@worker_process_init.connect()
def on_worker_init(**_):
    init()
    init_inpainting()
    print('----- INIT done !')


@cel_app.task(name='txt2img')
def txt2img(params=None):
    image, path = run_txt2img(model, params)
    return path

@cel_app.task(name='inpaint')
def inpaint(opt=None):
    from types import SimpleNamespace
    opt = SimpleNamespace(**opt)

    os.makedirs(opt.outdir, exist_ok=True)

    output_image = run_inpaint(inpainting_model, opt.image_path, 30)

    outpath = os.path.join(opt.outdir, f'{opt.file_prefix}-0000.png')
    output_image.save(outpath)
    return 'OK'
