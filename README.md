# ComfyUI Krea 2 Depth Control

One node, **Krea 2 Depth Control** (`Krea2DepthControl`): guides a Krea 2 render by a depth map,
with a depth control LoRA such as [Patil/Krea-2-depth-controlnet](https://huggingface.co/Patil/Krea-2-depth-controlnet)
(weights under the Krea 2 Community License; this code is MIT).

Inputs: the model, the Krea 2 / Qwen-Image VAE, the depth map (inverse depth, near white), the
sampler's latent (the map is fitted to its size), the control LoRA from `models/loras`, and a
strength (0.75 is a good start; lower is looser). Put the returned model into the sampler.

Works beside reference-latent nodes (pose, edit): their tokens get no depth.
