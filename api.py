# api.py

# @dev Scope: Trains LORA using Kohya sd-scripts

import os
import uuid
import shutil
import subprocess
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional  # Add Optional here
from datetime import datetime
import json
import asyncio
from google.cloud import storage

# IPFS Gateway Configuration
IPFS_GATEWAYS = [
    "https://gateway.pinata.cloud/ipfs/",
    "https://ipfs.io/ipfs/",
    "https://cloudflare-ipfs.com/ipfs/",
    "https://gateway.ipfs.io/ipfs/"
]

app = FastAPI()

# Define the request body schema
class TrainParams(BaseModel):
    subscriptionId: str
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

async def download_from_ipfs(cid: str, file_path: str) -> bool:
    """Download a file from IPFS using multiple gateways with retry logic."""
    for gateway in IPFS_GATEWAYS:
        try:
            url = f"{gateway}{cid}"
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
        except Exception as e:
            print(f"Failed to download from {gateway}: {str(e)}")
            continue
    return False

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

        # Handle IPFS dataset
        if params.dataset_source == "ipfs":
            for cid in params.cids:
                file_path = os.path.join(dataset_dir, f"{cid}.png")
                caption_path = os.path.join(dataset_dir, f"{cid}.caption")
                try:
                    if not os.path.isfile(file_path):
                        success = asyncio.run(download_from_ipfs(cid, file_path))
                        if not success:
                            raise HTTPException(status_code=500, detail=f"Failed to download CID {cid} from all gateways")

                    tags = params.tags.get(cid, None)
                    if tags:
                        with open(caption_path, "w") as caption_file:
                            caption_file.write(tags)
                except Exception as error:
                    raise HTTPException(status_code=500, detail=f"Error processing CID {cid}: {error}")

        # Handle GCS dataset
        elif params.dataset_source == "gcs":
            try:
                storage_client = storage.Client()
                bucket_name = "privateuploads"
                dataset_prefix = f"user/{params.wallet_address}/{params.dataset_name}/"

                bucket = storage_client.bucket(bucket_name)
                blobs = bucket.list_blobs(prefix=dataset_prefix)

                for blob in blobs:
                    if blob.name.endswith("/"):
                        continue  # Skip directory placeholders

                    file_name = blob.name.split("/")[-1]
                    local_path = os.path.join(dataset_dir, file_name)

                    print(f"Downloading {blob.name} to {local_path}...")
                    blob.download_to_filename(local_path)

                    # 🔹 Handle `.caption` files (Updated Logic)
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        caption_blob_name = f"{blob.name}.caption"
                        caption_blob = bucket.blob(caption_blob_name)
                        caption_path = os.path.join(dataset_dir, f"{file_name}.caption")

                        # ✅ If caption file exists in GCS, download it
                        if caption_blob.exists():
                            print(f"Downloading caption {caption_blob_name} to {caption_path}...")
                            caption_blob.download_to_filename(caption_path)
                        # ✅ Otherwise, check if a caption is provided in `params.tags`
                        elif file_name in params.tags:
                            print(f"Writing caption for {file_name}...")
                            with open(caption_path, "w") as caption_file:
                                caption_file.write(params.tags[file_name])

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error downloading dataset from GCS: {str(e)}")

        else:
            raise HTTPException(status_code=400, detail="Invalid dataset_source. Use 'ipfs' or 'gcs'.")

        # ✅ Generate dataset config
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
min_bucket_reso = 512

  [[datasets.subsets]]
  image_dir = "{dataset_dir}"
  caption_dropout_rate = 0.0
""")

        # Prepare subdirectory for this model
        model_subdir = os.path.join(model_dir, params.model_name)
        os.makedirs(model_subdir, exist_ok=True)

        # Prepare model output path
        model_output_path = os.path.join(model_subdir, f"{params.model_name}.safetensors")

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
            "--output_dir", model_subdir, # needs to be persistent per user
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

        # Prepare full training metadata to store as JSON
        params_dict = {
            "model_name": params.model_name,
            "dataset_name": params.dataset_name,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "wallet_address": params.wallet_address,
            "cids": params.cids,
            "tags": params.tags,
            "command": command  # full raw list form of CLI command used
        }

        # Remove keys with empty values
        params_dict = {k: v for k, v in params_dict.items() if v not in [None, [], {}]}

        # Write metadata to file
        params_path = f"{model_subdir}/{params.model_name}.json"
        with open(params_path, "w") as f:
            json.dump(params_dict, f, indent=2)

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
    model_subdir = f"/workspace/project/sd-scripts/user/{wallet_address}/models/{model_name}"
    model_file = f"{model_subdir}/{model_name}.safetensors"
    param_file = f"{model_subdir}/{model_name}.json"

    gcs_destination_model = f"gs://lorabucketnorepi/user/{wallet_address}/models/{model_name}/{model_name}.safetensors"
    gcs_destination_param = f"gs://lorabucketnorepi/user/{wallet_address}/models/{model_name}/{model_name}.json"

    # Upload model and metadata JSON
    subprocess.run([
        "gsutil", "-m", "cp", "-r",
        model_subdir,
        f"gs://lorabucketnorepi/user/{wallet_address}/models/"
    ], check=True)


@app.get("/train/status/{job_id}")
def check_status(job_id: str):
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return job_status[job_id]


ready_event = asyncio.Event()

@app.on_event("startup")
async def on_startup():
    # Signal that the app is ready
    ready_event.set()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

async def notify_backend():
    try:
        NEXTJS_BACKEND_URL = os.getenv("NEXTJS_BACKEND_URL", "https://www.wispi.art")
        SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")  # Get from env but don't default

        # Wait for FastAPI to be truly ready
        await ready_event.wait()
        await asyncio.sleep(5)  # Additional safety delay

        print(f"📣 Notifying backend at {NEXTJS_BACKEND_URL}/api/runpod/markReady")

        response = requests.post(f"{NEXTJS_BACKEND_URL}/api/runpod/markReady", json={
            "subscription_id": SUBSCRIPTION_ID,
            "status": "ready",
            "runtime": "lora-container-runtime",
        })

        if response.status_code == 200:
            print("✅ Successfully notified backend.")
        else:
            print(f"⚠️ Failed to notify backend: {response.status_code} - {response.text}")
    except Exception as e:
        print("❌ Error notifying backend:", e)

# Start the notification process
@app.on_event("startup")
async def start_notification():
    asyncio.create_task(notify_backend())
