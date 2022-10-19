# This file is part of Imagine server.

# Imagine server is free software: you can redistribute it and / or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# Imagine server is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# A copy of the GNU General Public License is provided in the "COPYING" file. If not, see https:// www.gnu.org/licenses/

import torch
from numpy import float16
from numpy import array as np_array
from PIL import Image, ImageFilter
from PIL.PngImagePlugin import PngInfo

torch.set_default_tensor_type(torch.HalfTensor)

#https://www.metadata2go.com/
def metadata (opt, prompt = None, seed = '', generation_time = None):
    data = PngInfo()
    if (prompt is not None):
        data.add_text('prompt',   prompt)
    if (generation_time is not None):
        data.add_text('generation time', str(generation_time))

    data.add_text('seed',     str(seed))
    data.add_text('steps',    str(opt.steps))
    data.add_text('guidance', str(opt.scale))
    if opt.image_guide:
        if opt.blend_mask is None:
            data.add_text('with guided image and strength', str(opt.strength))
        else:
            data.add_text('with guided image', 'and blend mask')
    data.add_text('model', 'Toxicode DreamServer - Stable Diffusion 1.4')
    return data



def make_inpaint_batch(image_path, device):
    image = np_array(Image.open(image_path))
    mask  = image[:,:,3]
    image = image[:,:,0:3]
    image = image.astype(float16)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)
    mask  = mask[None,None]
    mask[mask <= 100] = 1
    mask[mask > 100]  = 0
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].half().to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch



#FIXME optimize ?
def get_mask_for_latent_blending(device, path, blur = 0):
    mask_image = Image.open(path).convert("L")

    if blur > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(blur))

    mask_for_reconstruction = mask_image.point(lambda x: 255 if x > 0 else 0)
    mask_for_reconstruction = mask_for_reconstruction.filter(
        ImageFilter.GaussianBlur(radius=10))
    mask_for_reconstruction = mask_for_reconstruction.point(
        lambda x: 255 if x > 127 else x * 2)

    mask_for_reconstruction = torch.from_numpy(
        (np_array(mask_for_reconstruction) / 255.0).astype(float16)).to(device)

    source_w, source_h = mask_image.size


    mask = np_array(
        mask_image.resize(
            (int(source_w / 8), int(source_h / 8)), resample=Image.Resampling.LANCZOS).convert("L"))
    mask = (mask / 255.0).astype(float16)

    mask = mask[None]
    mask = 1 - mask

    mask = torch.from_numpy(mask)

    mask = torch.stack([mask, mask, mask, mask], 1).to(device)  # FIXME
    return [mask_for_reconstruction, mask]
