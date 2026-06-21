import nodes
import node_helpers
import torch
import torchaudio
import comfy.model_management
import comfy.model_sampling
import comfy.samplers
import comfy.utils
import math
import numpy as np
import av
from io import BytesIO
from typing_extensions import override
from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier, latent_to_pixel_coords
from comfy_api.latest import ComfyExtension, io


def get_noise_mask(latent):
    noise_mask = latent.get("noise_mask", None)
    latent_image = latent["samples"]
    if noise_mask is None:
        batch_size, _, latent_length, _, _ = latent_image.shape
        noise_mask = torch.ones(
            (batch_size, 1, latent_length, 1, 1),
            dtype=torch.float32,
            device=latent_image.device,
        )
    else:
        noise_mask = noise_mask.clone()
    return noise_mask


class LTXVImgToVideoStartEnd(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVImgToVideoStartEnd",
            category="model/conditioning/ltxv",
            inputs=[
                io.Vae.Input("vae"),
                io.Image.Input("image_first"),
                # option specify the last frame
                io.Image.Input("image_last", optional=True),
                io.Latent.Input("latent"),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0),
                io.Boolean.Input("bypass", default=False, tooltip="Bypass the conditioning."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, vae, image_first, latent, strength, bypass=False, image_last=None) -> io.NodeOutput:
        if bypass:
            return (latent,)

        samples = latent["samples"].clone()
        _, height_scale_factor, width_scale_factor = (
            vae.downscale_index_formula
        )

        _, _, _, latent_height, latent_width = samples.shape
        width = latent_width * width_scale_factor
        height = latent_height * height_scale_factor

        # Helper function to resize and encode an image to latent space
        def process_and_encode(img):
            if img.shape[1] != height or img.shape[2] != width:
                pixels = comfy.utils.common_upscale(
                    img.movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
            else:
                pixels = img
            encode_pixels = pixels[:, :, :, :3]
            return vae.encode(encode_pixels)

        # 1. Process and inject the first frame
        t_first = process_and_encode(image_first)
        samples[:, :, :t_first.shape[2]] = t_first

        # 2. Grab initial noise mask
        conditioning_latent_frames_mask = get_noise_mask(latent)
        
        # Mask the first frame(s)
        conditioning_latent_frames_mask[:, :, :t_first.shape[2]] = 1.0 - strength

        # 3. Process and inject the last frame if provided
        if image_last is not None:
            t_last = process_and_encode(image_last)
            
            # Target the trailing frames via negative slicing
            samples[:, :, -t_last.shape[2]:] = t_last
            
            # Mask the trailing frames
            conditioning_latent_frames_mask[:, :, -t_last.shape[2]:] = 1.0 - strength

        return io.NodeOutput({"samples": samples, "noise_mask": conditioning_latent_frames_mask})

    generate = execute  # TODO: remove

NODE_CLASS_MAPPINGS = {
    "LTXVImgToVideoStartEnd": LTXVImgToVideoStartEnd,
}
