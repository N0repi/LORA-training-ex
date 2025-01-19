# api.py

# @dev Scope: Trains LORA using Kohya sd-scripts

import os
import shutil
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional  # Add Optional here
from datetime import datetime
from ipfs_api import download

app = FastAPI()

# Define the request body schema
class TrainParams(BaseModel):
    wallet_address: str
    model_name: str = Field(..., min_length=1)
    network_dim: int
    learning_rate: float
    max_train_epochs: int
    max_train_steps: int
    cids: List[str] = Field(..., min_items=1)
    tags: Optional[dict] = {}  # Optional field, defaults to an empty dictionary

@app.post("/train")
def train_model(params: TrainParams):
    try:
        # Base directories
        base_dir = f"/workspace/project/sd-scripts/user{params.wallet_address}"
        config_dir = f"{base_dir}/configs"
        model_dir = f"{base_dir}/models"
        dataset_dir = f"{base_dir}/datasets"
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)

        # Download images using IPFS CIDs and create tags if exist
        for cid in params.cids:
            file_path = os.path.join(dataset_dir, f"{cid}.png")
            caption_path = os.path.join(dataset_dir, f"{cid}.caption")
            try:
                # Download the image using IPFS
                if not os.path.isfile(file_path):
                    download(cid, file_path)

                # Write the caption file only if tags are provided
                tags = params.tags.get(cid, None)
                if tags:
                    with open(caption_path, "w") as caption_file:
                        caption_file.write(tags)
            except Exception as error:
                raise HTTPException(status_code=500, detail=f"Error processing CID {cid}: {error}")


        # Create a single TOML configuration file
        config_path = f"{config_dir}/dataset_config.toml"
        with open(config_path, "w") as toml_file:
            toml_file.write(f"""
[general]
shuffle_caption = false
caption_extension = '.caption'
keep_tokens = 1

[[datasets]]
resolution = [1024, 1024]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false
bucket_reso_steps = 64
min_bucket_reso = 768

  [[datasets.subsets]]
  image_dir = "{dataset_dir}"
  caption_dropout_rate = 0.0
""")

        # Prepare model output path
        model_output_path = f"{model_dir}/{params.model_name}.safetensors"

        # Command to execute
        command = [
            "accelerate", "launch",
            "--mixed_precision", "bf16",
            "--num_cpu_threads_per_process", "1",
            "flux_train_network.py",
            "--pretrained_model_name_or_path", "/workspace/project/sd-scripts/models/flux_devFp8.safetensors",
            "--clip_l", "/workspace/project/sd-scripts/models/clip_l.safetensors",
            "--t5xxl", "/workspace/project/sd-scripts/models/t5xxl_fp16.safetensors",
            "--ae", "/workspace/project/sd-scripts/models/ae.safetensors",
            "--cache_latents_to_disk",
            "--cache_text_encoder_outputs_to_disk",
            "--save_model_as", "safetensors",
            "--sdpa",
            "--persistent_data_loader_workers",
            "--max_data_loader_n_workers", "2",
            "--seed", "42",
            "--gradient_checkpointing",
            "--mixed_precision", "bf16",
            "--save_precision", "bf16",
            "--network_module", "networks.lora_flux",
            "--network_dim", str(params.network_dim),  # Dynamic
            "--network_train_unet_only",
            "--learning_rate", str(params.learning_rate),  # Dynamic
            "--cache_text_encoder_outputs",
            "--cache_text_encoder_outputs_to_disk",
            "--fp8_base",
            "--highvram",
            "--max_train_epochs", str(params.max_train_epochs),  # Dynamic
            "--max_train_steps", str(params.max_train_steps),  # Dynamic
            "--dataset_config", config_path, # needs to be persistent per user
            "--output_dir", model_dir, # needs to be persistent per user
            "--output_name", params.model_name,
            "--timestep_sampling", "shift",
            "--discrete_flow_shift", "3.1582",
            "--optimizer_type", "adafactor",
            "--optimizer_args", "relative_step=False", "scale_parameter=False", "warmup_init=False",
            "--model_prediction_type", "raw",
            "--guidance_scale", "1.0",
            "--fused_backward_pass",
            "--blocks_to_swap", "4",
            "--full_bf16"
        ]

        # Execute training
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Training failed: {stderr}")

        if process.returncode != 0:
            print("Training failed. Command output:")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            raise HTTPException(
                status_code=500,
                detail=f"Training failed: {stderr}"
            )

        return {
            "message": "Training completed successfully",
            "model_path": model_output_path,
            "stdout": stdout,
            "stderr": stderr,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
