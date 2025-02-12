# api.py

# @dev Scope: Trains LORA using Kohya sd-scripts

import os
import uuid
import shutil
import subprocess
from fastapi import FastAPI, HTTPException, BackgroundTasks
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
    dataset_source: str  # Either "ipfs" or "gcs"
    dataset_name: Optional[str] = None  # Used only for GCS
    cids: Optional[List[str]] = Field(default_factory=list)
    tags: Optional[dict] = {}  # Optional field, defaults to an empty dictionary

job_status = {}
@app.post("/train")
def train_model(params: TrainParams, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job_status[job_id] = {"status": "in_progress", "wallet_address": params.wallet_address}
    background_tasks.add_task(run_training_and_sync, params, job_id)
    return {"message": "Training started", "job_id": job_id}


def run_training_and_sync(params, job_id):
    try:
        # Base directories
        base_dir = f"/workspace/project/sd-scripts/user/{params.wallet_address}"
        config_dir = f"{base_dir}/configs"
        model_dir = f"{base_dir}/models"
        dataset_dir = f"{base_dir}/datasets"
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)

        if params.dataset_source == "ipfs":
            # 🔹 Download images using IPFS (same as existing logic)
            for cid in params.cids:
                file_path = os.path.join(dataset_dir, f"{cid}.png")
                caption_path = os.path.join(dataset_dir, f"{cid}.caption")
                try:
                    if not os.path.isfile(file_path):
                        download(cid, file_path)

                    tags = params.tags.get(cid, None)
                    if tags:
                        with open(caption_path, "w") as caption_file:
                            caption_file.write(tags)
                except Exception as error:
                    raise HTTPException(status_code=500, detail=f"Error processing CID {cid}: {error}")

        elif params.dataset_source == "gcs":
            # 🔹 Download dataset from GCS
            storage_client = storage.Client()
            bucket_name = "privateuploads"
            dataset_prefix = f"user/{params.wallet_address}/{params.dataset_name}/"

            bucket = storage_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=dataset_prefix)

            for blob in blobs:
                # Skip directory placeholder files
                if blob.name.endswith("/"):
                    continue

                file_name = blob.name.split("/")[-1]  # Extract the filename
                local_path = os.path.join(dataset_dir, file_name)

                print(f"Downloading {blob.name} to {local_path}...")
                blob.download_to_filename(local_path)

        else:
            raise HTTPException(status_code=400, detail="Invalid dataset_source. Use 'ipfs' or 'gcs'.")

        # Create TOML configuration file (same as before)
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
            "--num_cpu_threads_per_process", "1",
            "/workspace/project/sd-scripts/flux_train_network.py",
            "--pretrained_model_name_or_path", "/workspace/project/sd-scripts/models/flux1-dev-fp8.safetensors",
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
            "--mixed_precision", "fp16",
            "--save_precision", "fp16",
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
        ]
        # --full_bf16

        # Execute training
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = stderr
            return

        # Delete datasets folder after training to save storage
        shutil.rmtree(dataset_dir, ignore_errors=True)

        job_status[job_id]["status"] = "uploading"
        sync_to_gcs(params.wallet_address, params.model_name)
        job_status[job_id]["status"] = "completed"
        job_status[job_id]["model_path"] = f"user/{params.wallet_address}/models/{params.model_name}.safetensors"

    except Exception as e:
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = str(e)


def sync_to_gcs(wallet_address, model_name):
    model_file = f"/workspace/project/sd-scripts/user/{wallet_address}/models/{model_name}.safetensors"
    gcs_destination = f"gs://lorabucketnorepi/user/{wallet_address}/models/{model_name}.safetensors"

    # Upload only models folder directly to the user's bucket directory
    command = [
        "gsutil", "cp",
        model_file,
        gcs_destination
    ]

    subprocess.run(command, check=True)


@app.get("/train/status/{job_id}")
def check_status(job_id: str):
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return job_status[job_id]
