# trainLora.py

# run server: uvicorn api:app --host 0.0.0.0 --port 8000
# call: curl -X POST http://192.168.3.47:8000/train

import subprocess
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/train")
def train_model():
    try:
        # Command to execute
        command = [
            "accelerate", "launch",
            "--mixed_precision", "bf16",
            "--num_cpu_threads_per_process", "1",
            "flux_train_network.py",
            "--pretrained_model_name_or_path", "/mnt/NAS/AI-ML/SD/SD_newer/torch1/new-install/forge/stable-diffusion-webui-forge/models/Stable-diffusion/flux/flux_devFp8.safetensors",
            "--clip_l", "/mnt/NAS/AI-ML/SD/SD_newer/torch1/new-install/forge/training/ai-toolkit/models/clip_l.safetensors",
            "--t5xxl", "/mnt/NAS/AI-ML/SD/SD_newer/torch1/new-install/forge/training/ai-toolkit/models/t5xxl_fp16.safetensors",
            "--ae", "/mnt/NAS/AI-ML/SD/SD_newer/torch1/new-install/forge/training/ai-toolkit/models/ae.safetensors",
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
            "--network_dim", "4",
            "--network_train_unet_only",
            "--learning_rate", "1e-5",
            "--cache_text_encoder_outputs",
            "--cache_text_encoder_outputs_to_disk",
            "--fp8_base",
            "--highvram",
            "--max_train_epochs", "143",
            "--save_every_n_epochs", "4",
            "--max_train_steps", "4000",
            "--dataset_config", "configs/YukiTrain2.toml",
            "--output_dir", "outputs/YukiTrain2",
            "--output_name", "flux-lora-name",
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

        # Execute the command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        # Check for errors
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Training failed: {stderr}")

        return {"message": "Training started successfully", "output": stdout}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

