import argparse
import time
from huggingface_hub import snapshot_download

def get_args():
    parser = argparse.ArgumentParser(description="HuggingFace Repository Downloader")
    parser.add_argument('--ak', type=str, required=True, help='Your HuggingFace Access Key / Token')
    return parser.parse_args()

def download_with_retry(repo_id, local_dir, cache_dir, ak, ignore_patterns=None, allow_patterns=None):
    print(f"Start downloading: {repo_id} -> {local_dir}")
    while True:
        try:
            snapshot_download(
                cache_dir=cache_dir,
                local_dir=local_dir,
                repo_id=repo_id,
                local_dir_use_symlinks=False,
                resume_download=True,
                token=ak,
                ignore_patterns=ignore_patterns,
                allow_patterns=allow_patterns,
            )
            print(f"Successfully downloaded: {repo_id}")
            break
        except Exception as e:
            print(f"Download failed for {repo_id}, retrying in 5 seconds... Error: {e}")
            time.sleep(5)
            continue

if __name__ == "__main__":
    args = get_args()
    ak = args.ak

    base_cache_path = './checkpoints/'

    # --- Task 1: hunyuanvideo ---
    repo_id   = 'hunyuanvideo-community/HunyuanVideo'
    save_dir  = repo_id
    cache_dir = base_cache_path + save_dir + "/cache"
    
    download_with_retry(
        repo_id=repo_id,
        local_dir=save_dir,
        cache_dir=cache_dir,
        ak=ak,
        ignore_patterns=["text_encoder/model*", "transformer/diffusion*"] 
    )

    # --- Task 2: qwen3vl ---
    repo_id   = 'Qwen/Qwen3-VL-4B-Instruct'
    save_dir  = repo_id
    cache_dir = base_cache_path + save_dir + "/cache"

    download_with_retry(
        repo_id=repo_id,
        local_dir=save_dir,
        cache_dir=cache_dir,
        ak=ak
    )

    # --- Task 3: VINO ---
    repo_id   = 'SOTAMak1r/VINO-weight'
    save_dir  = repo_id
    cache_dir = base_cache_path + save_dir + "/cache"

    download_with_retry(
        repo_id=repo_id,
        local_dir=save_dir,
        cache_dir=cache_dir,
        ak=ak,
        allow_patterns=['vino.ckpt',]
    )