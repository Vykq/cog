# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[
                "epicrealism_naturalSinRC1VAE.safetensors",
                "ip-adapter-plus_sd15.bin",
                "clip-vision-large.safetensors"
            ],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    # Update nodes in the JSON workflow to modify your workflow based on the given inputs
    def update_workflow(self, workflow, **kwargs):
        # Update the prompt in node 6
        if "prompt" in kwargs and kwargs["prompt"]:
            positive_prompt = workflow["6"]["inputs"]
            positive_prompt["text"] = kwargs["prompt"]

        # Update the negative prompt in node 7
        if "negative_prompt" in kwargs and kwargs["negative_prompt"]:
            negative_prompt = workflow["7"]["inputs"]
            negative_prompt["text"] = kwargs["negative_prompt"]

        # Update the seed in the KSampler node 3
        if "seed" in kwargs:
            sampler = workflow["3"]["inputs"]
            sampler["seed"] = kwargs["seed"]
        
        # Update the input image in node 33 if provided
        if "image_filename" in kwargs and kwargs["image_filename"]:
            image_node = workflow["33"]["inputs"]
            # Use a local file path for the image
            image_path = os.path.join(INPUT_DIR, kwargs["image_filename"])
            image_node["image"] = image_path

    def predict(
        self,
        prompt: str = Input(
            description="Your positive prompt",
            default="commercial photo, light green and white, greenery background, depth of field, high level feeling, perfect lighting, OC renderer, Blender, super sharp, super noise reduction",
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image",
            default="(noise, blur, worst quality, low quality, error, cropped, bad anatomy, bad proportions, wrong hands) (NSFW, nude)",
        ),
        image: Path = Input(
            description="Input image for the generation (will be used as the base image)",
            default=None,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # Make sure to set the seeds in your workflow
        seed = seed_helper.generate(seed)

        image_filename = None
        if image:
            image_filename = self.filename_with_extension(image, "image")
            self.handle_input_file(image, image_filename)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_filename=image_filename,
            seed=seed,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )
