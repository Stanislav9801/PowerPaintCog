import os
import torch
import numpy as np
from PIL import Image
from cog import BasePredictor, Input, Path
from transformers import CLIPTextModel
from diffusers import UniPCMultistepScheduler
from powerpaint.models.BrushNet_CA import BrushNetModel
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
from powerpaint.utils.utils import TokenizerWrapper, add_tokens
from safetensors.torch import load_model

class Predictor(BasePredictor):
    def setup(self):
        self.weight_dtype = torch.float16
        self.checkpoint_dir = "checkpoints/ppt-v2"
        self.local_files_only = True
        self.version = "ppt-v2"

        # Initialize PowerPaint v2 pipeline
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
            revision=None,
            torch_dtype=self.weight_dtype,
            local_files_only=self.local_files_only,
        )
        text_encoder_brushnet = CLIPTextModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="text_encoder",
            revision=None,
            torch_dtype=self.weight_dtype,
            local_files_only=self.local_files_only,
        )
        brushnet = BrushNetModel.from_unet(unet)
        base_model_path = os.path.join(self.checkpoint_dir, "realisticVisionV60B1_v51VAE")
        
        self.pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
            base_model_path,
            brushnet=brushnet,
            text_encoder_brushnet=text_encoder_brushnet,
            torch_dtype=self.weight_dtype,
            low_cpu_mem_usage=False,
            safety_checker=None,
        )
        self.pipe.unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
            revision=None,
            torch_dtype=self.weight_dtype,
            local_files_only=self.local_files_only,
        )
        self.pipe.tokenizer = TokenizerWrapper(
            from_pretrained=base_model_path,
            subfolder="tokenizer",
            revision=None,
            torch_type=self.weight_dtype,
            local_files_only=self.local_files_only,
        )

        # Add learned task tokens into the tokenizer
        add_tokens(
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder_brushnet,
            placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
            initialize_tokens=["a", "a", "a"],
            num_vectors_per_token=10,
        )
        
        load_model(
            self.pipe.brushnet,
            os.path.join(self.checkpoint_dir, "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"),
        )

        self.pipe.text_encoder_brushnet.load_state_dict(
            torch.load(os.path.join(self.checkpoint_dir, "PowerPaint_Brushnet/pytorch_model.bin")), strict=False
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        self.pipe.enable_model_cpu_offload()
        self.pipe = self.pipe.to("cuda")

    def predict(
        self,
        input_image: Path = Input(description="Input image"),
        mask_image: Path = Input(description="Mask image"),
        prompt: str = Input(description="Prompt for generation"),
        negative_prompt: str = Input(description="Negative prompt", default=""),
        task: str = Input(description="Task type", choices=["inpainting", "text-guided", "object-removal", "shape-guided", "image-outpainting"], default="inpainting"),
        fitting_degree: float = Input(description="Fitting degree", default=1.0, ge=0.0, le=1.0),
        num_inference_steps: int = Input(description="Number of denoising steps", default=45, ge=1, le=100),
        guidance_scale: float = Input(description="Guidance scale", default=7.5, ge=0.1, le=30.0),
        seed: int = Input(description="Random seed", default=None),
    ) -> Path:
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        else:
            generator = None

        input_image = Image.open(input_image).convert("RGB")
        mask_image = Image.open(mask_image).convert("RGB")

        # Resize images
        size1, size2 = input_image.size
        if size1 < size2:
            input_image = input_image.resize((640, int(size2 / size1 * 640)))
        else:
            input_image = input_image.resize((int(size1 / size2 * 640), 640))

        img = np.array(input_image)
        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
        input_image = input_image.resize((H, W))
        mask_image = mask_image.resize((H, W))

        promptA, promptB, negative_promptA, negative_promptB = self.add_task(prompt, negative_prompt, task, self.version)

        # Prepare input image and mask
        np_inpimg = np.array(input_image)
        np_inmask = np.array(mask_image) / 255.0
        np_inpimg = np_inpimg * (1 - np_inmask)
        input_image = Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB")

        result = self.pipe(
            promptA=promptA,
            promptB=promptB,
            promptU=prompt,
            tradoff=fitting_degree,
            tradoff_nag=fitting_degree,
            image=input_image,
            mask=mask_image,
            num_inference_steps=num_inference_steps,
            generator=generator,
            brushnet_conditioning_scale=1.0,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            negative_promptU=negative_prompt,
            guidance_scale=guidance_scale,
            width=H,
            height=W,
        ).images[0]

        output_path = "/tmp/output.png"
        result.save(output_path)
        return Path(output_path)

    def add_task(self, prompt, negative_prompt, task, version):
        if task == "object-removal":
            prompt = prompt + " empty scene blur"
        elif task == "image-outpainting":
            prompt = prompt + " empty scene"
        elif task == "inpainting":
            # For inpainting, we don't modify the prompt
            pass
        
        promptA = prompt + " P_obj"
        promptB = prompt + " P_obj"
        negative_promptA = negative_prompt + " P_obj"
        negative_promptB = negative_prompt + " P_obj"

        return promptA, promptB, negative_promptA, negative_promptB