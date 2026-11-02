"""Depth control for Krea 2 in ComfyUI.

Runs a depth control LoRA trained the way Patil/Krea-2-depth-controlnet describes: the depth map
is VAE-encoded, normalized like the sampler's latent, patchified like the image, and its tokens
are concatenated with the image tokens on the channel dim before the DiT's input projection,
which the checkpoint replaces with a 2x-wide one (`first.weight` [features, 2 * 64]). The
transformer blocks get the checkpoint's rank-r LoRA pairs (`blocks.N.<path>.A` / `.B`, alpha =
rank) through ComfyUI's own LoRA patching, so offloading and other LoRAs work as usual.

Reference-image tokens a pose or edit node appends after the image's own get zero control.
"""

import torch
import torch.nn.functional as F

import comfy.lora
import comfy.utils
import comfy.patcher_extension
import folder_paths

WRAPPER = "krea2_depth_control"


def _split(state):
    """The checkpoint as ComfyUI LoRA keys for the blocks, and the wide input projection."""
    lora = {}
    for key, value in state.items():
        if key.startswith("blocks.") and key[-2:] in (".A", ".B"):
            name = "diffusion_model." + key[:-2]
            lora[name + (".lora_down.weight" if key.endswith(".A") else ".lora_up.weight")] = value
    if "first.weight" not in state:
        raise RuntimeError("Not a Krea 2 control LoRA: it has no first.weight (the wide input projection).")
    return lora, state["first.weight"], state.get("first.bias")


class _WideFirst(torch.nn.Module):
    """The input projection with the control tokens beside the image's."""

    def __init__(self, base, weight, bias, control):
        super().__init__()
        self.base, self.weight, self.bias, self.control = base, weight, bias, control

    def forward(self, img):
        ctrl = self.control.to(img.device, img.dtype)
        ctrl = comfy.utils.repeat_to_batch_size(ctrl, img.shape[0])
        extra = img.shape[1] - ctrl.shape[1]
        if extra > 0:  # reference tokens follow the image's: no control for them
            ctrl = torch.cat([ctrl, ctrl.new_zeros(img.shape[0], extra, ctrl.shape[2])], dim=1)
        bias = self.bias if self.bias is not None else self.base.bias
        return F.linear(torch.cat([img, ctrl[:, : img.shape[1]]], dim=-1), self.weight.to(img.device, img.dtype),
                        None if bias is None else bias.to(img.device, img.dtype))


def _wrapper(weight, bias, latent):
    def wrap(executor, x, *args, **kwargs):
        dm = executor.class_obj
        lat = latent
        if lat.shape[-2:] != x.shape[-2:]:
            lat = F.interpolate(lat.float(), size=x.shape[-2:], mode="bilinear", align_corners=False)
        tokens = dm.process_img(lat.to(x.device, x.dtype))[0]
        base = dm.first
        dm.first = _WideFirst(base, weight, bias, tokens)
        try:
            return executor(x, *args, **kwargs)
        finally:
            dm.first = base

    return wrap


class Krea2DepthControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "depth": ("IMAGE", {"tooltip": "Inverse depth: near white, far black."}),
                "latent": ("LATENT", {"tooltip": "The sampler's latent: the depth map is fitted to its size."}),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Krea2"

    def apply(self, model, vae, depth, latent, lora_name, strength):
        if strength == 0:
            return (model,)
        state = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True)
        lora, weight, bias = _split(state)

        # The depth map at the latent's pixel size, grey, near white (as Depth-Anything gives it).
        samples = latent["samples"]
        h, w = samples.shape[-2] * 8, samples.shape[-1] * 8
        img = depth[..., :3].movedim(-1, 1)
        # Centre-cropped to the render's shape first, so a map made for another shape isn't stretched.
        ih, iw = img.shape[-2:]
        if abs(iw / ih - w / h) > 0.01:
            cw, ch = (round(ih * w / h), ih) if iw / ih > w / h else (iw, round(iw * h / w))
            top, left = (ih - ch) // 2, (iw - cw) // 2
            img = img[..., top:top + ch, left:left + cw]
        img = F.interpolate(img, size=(h, w), mode="bilinear", align_corners=False)
        img = img.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1).movedim(1, -1).clamp(0, 1)
        encoded = vae.encode(img)
        if encoded.ndim == 4:
            encoded = encoded.unsqueeze(2)
        control = model.model.process_latent_in(encoded)[:, :, 0]

        patched = model.clone()
        keys = comfy.lora.model_lora_keys_unet(patched.model, {})
        patches = comfy.lora.load_lora(lora, keys)
        if not patched.add_patches(patches, strength):
            raise RuntimeError("The model took none of the control LoRA's block weights: is it Krea 2?")
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER, _wrapper(weight, bias, control))
        return (patched,)


NODE_CLASS_MAPPINGS = {"Krea2DepthControl": Krea2DepthControl}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2DepthControl": "Krea 2 Depth Control"}
