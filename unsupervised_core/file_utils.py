import multiprocessing as mp
import pickle as pkl
from pathlib import Path
from tqdm import tqdm

def load_prediction_worker(args):
    """Worker function to load a single prediction file"""
    camera_name, cam_timestamp_ns, owlvit_pred_dir = args
    
    owlvit_pred_path = owlvit_pred_dir / f"{cam_timestamp_ns}_{camera_name}.pkl"
    
    try:
        with open(owlvit_pred_path, "rb") as f:
            prediction = pkl.load(f)
        
        pred_boxes = prediction["pred_boxes"]
        image_class_embeds = prediction["image_class_embeds"]
        objectness_scores = prediction["objectness_scores"]
        
        return {
            'camera_name': camera_name,
            'cam_timestamp_ns': cam_timestamp_ns,
            'pred_boxes': pred_boxes,
            'image_class_embeds': image_class_embeds,
            'objectness_scores': objectness_scores,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'camera_name': camera_name,
            'cam_timestamp_ns': cam_timestamp_ns,
            'pred_boxes': None,
            'image_class_embeds': None,
            'objectness_scores': None,
            'success': False,
            'error': str(e)
        }

def load_predictions_parallel(sweep_camera_name_timestamps, owlvit_pred_dir, num_workers=None):
    """
    Load predictions in parallel using forking workers
    
    Args:
        sweep_camera_name_timestamps: List of (camera_name, cam_timestamp_ns) tuples
        owlvit_pred_dir: Path to directory containing prediction files
        num_workers: Number of worker processes (default: cpu_count())
    
    Returns:
        List of dictionaries containing loaded predictions with camera info
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Prepare arguments for worker function
    worker_args = [
        (camera_name, cam_timestamp_ns, owlvit_pred_dir)
        for camera_name, cam_timestamp_ns in sweep_camera_name_timestamps
    ]
    
    # Use fork method for multiprocessing
    ctx = mp.get_context('fork')
    
    results = []
    with ctx.Pool(num_workers) as pool:
        # Use imap for better memory efficiency with large datasets
        for result in tqdm(
            pool.imap(load_prediction_worker, worker_args),
            total=len(worker_args),
            desc="_get_vision_guided_clusters: loading predictions in parallel"
        ):
            results.append(result)
    
    # Report any failed loads
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print(f"Warning: {len(failed_results)} predictions failed to load")
        for failed in failed_results[:5]:  # Show first 5 failures
            print(f"  Failed: {failed['camera_name']} @ {failed['cam_timestamp_ns']}: {failed['error']}")
    
    return results