import modal

app = modal.App("marigold-inference-test")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1", "torchvision==0.19.1", "diffusers==0.31.0",
        "transformers==4.44.2", "huggingface_hub==0.25.2",
        "datasets<3.0.0", "Pillow", "numpy",
    )
)

vol = modal.Volume.from_name("marigoldoutput", create_if_missing=False)

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/output": vol},
    secrets=[modal.Secret.from_name("huggingface-secret2")],
)
def test():
    import os, torch, numpy as np
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from datasets import load_dataset
    from PIL import Image

    token = os.environ["HF_TOKEN"]
    device = "cuda"
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", token=token).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained("/output/unet_best").to(device).eval()
    ddim = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    ddim.set_timesteps(100, device=device)

    tok = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", token=token)
    txt = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", token=token).to(device)
    ids = tok([""], padding="max_length", max_length=tok.model_max_length, return_tensors="pt").input_ids.to(device)
    empty_embed = txt(ids)[0].detach()

    ds = load_dataset("ahd03/nyu-depth-controlnet", split="train")
    test_img = ds[100]["conditioning_image"].convert("RGB").resize((512, 512))
    img_arr = np.asarray(test_img, dtype=np.float32) / 127.5 - 1.0
    img_t = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        z_img = vae.encode(img_t).latent_dist.mode() * vae.config.scaling_factor
        z = torch.randn(z_img.shape, device=device)
        for t in ddim.timesteps:
            out = unet(torch.cat([z_img, z], dim=1), t, encoder_hidden_states=empty_embed).sample
            z = ddim.step(out, t, z).prev_sample
        depth = vae.decode(z / vae.config.scaling_factor).sample
        depth_img = ((depth.mean(dim=1).squeeze() + 1) / 2).clamp(0, 1).cpu().numpy()

    Image.fromarray((depth_img * 255).astype(np.uint8)).save("/output/test_result.png")
    test_img.save("/output/test_input.png")
    print("Saved test_input.png and test_result.png to volume")

@app.local_entrypoint()
def main():
    test.remote()