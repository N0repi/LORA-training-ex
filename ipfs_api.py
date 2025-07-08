import os
import requests
from typing import List
import time

# List of public IPFS gateways to try
IPFS_GATEWAYS = [
    "https://red-worthy-booby-89.mypinata.cloud/ipfs/",  # Primary gateway (Pinata)
    "https://gateway.pinata.cloud/ipfs/",  # Backup Pinata gateway
    "https://ipfs.io/ipfs/",  # Public IPFS gateway
    "https://cloudflare-ipfs.com/ipfs/",  # Cloudflare gateway
    "https://dweb.link/ipfs/"  # Protocol Labs gateway
]

def download(cid: str, output_path: str, max_retries: int = 3, timeout: int = 30) -> None:
    """
    Download a file from IPFS using multiple public gateways with retry logic.
    
    Args:
        cid: The IPFS CID to download
        output_path: Where to save the downloaded file
        max_retries: Maximum number of retries per gateway
        timeout: Timeout in seconds for each download attempt
    """
    if not cid:
        raise ValueError("CID cannot be empty")

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    last_error = None
    for gateway in IPFS_GATEWAYS:
        url = f"{gateway}{cid}"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout, stream=True)
                response.raise_for_status()
                
                # Write the file in chunks to handle large files
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return  # Success - exit the function
                
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
    
    # If we get here, all gateways failed
    raise Exception(f"Failed to download {cid} from all gateways. Last error: {str(last_error)}")

def download_multiple(cids: List[str], output_dir: str) -> dict:
    """
    Download multiple files from IPFS in parallel.
    
    Args:
        cids: List of IPFS CIDs to download
        output_dir: Directory to save the downloaded files
        
    Returns:
        dict: Mapping of CIDs to their download status (success/failure)
    """
    results = {}
    for cid in cids:
        output_path = os.path.join(output_dir, f"{cid}.png")
        try:
            download(cid, output_path)
            results[cid] = "success"
        except Exception as e:
            results[cid] = f"failed: {str(e)}"
    return results 