import argparse
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.ops import box_iou, nms
from transformers import (
    AutoProcessor,
    Owlv2ForObjectDetection,
    Owlv2ImageProcessor,
    OwlViTForObjectDetection,
    OwlViTProcessor,
)

ARGOVERSE_PROMPTS = {
    "REGULAR_VEHICLE": [
        "a car", "a passenger car", "a sedan", "an SUV", "a pickup truck", 
        "a van", "a passenger vehicle", "a personal vehicle"
    ],
    
    "PEDESTRIAN": [
        "a person walking", "a pedestrian", "a person standing", 
        "a person on foot", "a walking person", "a standing person"
    ],
    
    "BOLLARD": [
        "a traffic bollard", "a short post", "a road bollard", 
        "a traffic control post", "a barrier post"
    ],
    
    "CONSTRUCTION_CONE": [
        "an orange traffic cone", "a construction cone", "an orange cone", 
        "a traffic safety cone", "a road cone"
    ],
    
    "CONSTRUCTION_BARREL": [
        "an orange construction barrel", "a traffic barrel", "an orange barrel", 
        "a construction safety barrel", "a road work barrel"
    ],
    
    "STOP_SIGN": [
        "a red stop sign", "an octagonal stop sign", "a stop sign", 
        "a red octagonal sign", "a traffic stop sign"
    ],
    
    "BICYCLE": [
        "a bicycle", "a bike", "a two-wheeled bicycle", "a pedal bike"
    ],
    
    "LARGE_VEHICLE": [
        "a large vehicle", "a big truck", "a fire truck", "an RV", 
        "a recreational vehicle", "a large van", "an emergency vehicle"
    ],
    
    "WHEELED_DEVICE": [
        "a skateboard", "a scooter", "a segway", "a wheeled device", 
        "a golf cart", "a personal mobility device"
    ],
    
    "BUS": [
        "a city bus", "a public bus", "a transit bus", "a passenger bus", 
        "an urban bus"
    ],
    
    "BOX_TRUCK": [
        "a box truck", "a delivery truck", "a cube truck", "a moving truck", 
        "a cargo truck with box"
    ],
    
    "SIGN": [
        "a traffic sign", "a road sign", "a yield sign", "a speed limit sign", 
        "a directional sign", "a construction sign"
    ],
    
    "TRUCK": [
        "a delivery truck", "a UPS truck", "a FedEx truck", "a mail truck", 
        "a garbage truck", "a utility truck", "an ambulance", "a dump truck"
    ],
    
    "MOTORCYCLE": [
        "a motorcycle", "a motorbike", "a two-wheeled motorcycle"
    ],
    
    "BICYCLIST": [
        "a person riding a bicycle", "a cyclist", "a person on a bike", 
        "a bicycle rider", "a person cycling"
    ],
    
    "VEHICULAR_TRAILER": [
        "a trailer", "a vehicle trailer", "a towed trailer", "a cargo trailer"
    ],
    
    "TRUCK_CAB": [
        "a semi truck", "a tractor trailer cab", "a truck cab", 
        "a semi-trailer truck", "a big rig cab"
    ],
    
    "MOTORCYCLIST": [
        "a person riding a motorcycle", "a motorcyclist", "a person on a motorcycle", 
        "a motorcycle rider", "a biker"
    ],
    
    "DOG": [
        "a dog", "a canine", "a pet dog"
    ],
    
    "SCHOOL_BUS": [
        "a yellow school bus", "a school bus", "a student bus", 
        "a children's school bus"
    ],
    
    "WHEELED_RIDER": [
        "a person on a skateboard", "a person riding a scooter", 
        "a person on a wheeled device", "a skateboarder", "a scooter rider"
    ],
    
    "STROLLER": [
        "a baby stroller", "a pushchair", "a pram", "a child stroller"
    ],
    
    "ARTICULATED_BUS": [
        "an articulated bus", "a bendy bus", "a jointed bus", 
        "a long articulated bus"
    ],
    
    "MESSAGE_BOARD_TRAILER": [
        "an electronic message board", "a digital sign trailer", 
        "a LED message board", "a construction message sign"
    ],
    
    "MOBILE_PEDESTRIAN_SIGN": [
        "a pedestrian crossing sign", "a mobile crosswalk sign", 
        "a portable pedestrian sign", "a movable crossing sign"
    ],
    
    "WHEELCHAIR": [
        "a wheelchair", "a person in a wheelchair", "a mobility wheelchair", 
        "an electric wheelchair"
    ],
    
    "RAILED_VEHICLE": [
        "a train", "a trolley", "a subway train", "a rail vehicle", 
        "a tram", "a train car"
    ],
    
    "OFFICIAL_SIGNALER": [
        "a traffic controller", "a person directing traffic", 
        "a traffic officer", "a flagperson", "a construction worker directing traffic"
    ],
    
    "TRAFFIC_LIGHT_TRAILER": [
        "a portable traffic light", "a temporary traffic signal", 
        "a mobile traffic light", "a construction traffic light"
    ],
    
    "ANIMAL": [
        "an animal", "a large animal", "a wild animal", "a farm animal"
    ]
}

# Simple implementation for your current code structure
def get_simple_prompts():
    """Returns simple prompts similar to your current f'a {x}' format"""
    return {
        "REGULAR_VEHICLE": "a car",
        "PEDESTRIAN": "a person walking", 
        "BOLLARD": "a traffic bollard",
        "CONSTRUCTION_CONE": "an orange traffic cone",
        "CONSTRUCTION_BARREL": "an orange construction barrel", 
        "STOP_SIGN": "a red stop sign",
        "BICYCLE": "a bicycle",
        "LARGE_VEHICLE": "a large vehicle",
        "WHEELED_DEVICE": "a skateboard",
        "BUS": "a city bus",
        "BOX_TRUCK": "a box truck", 
        "SIGN": "a traffic sign",
        "TRUCK": "a delivery truck",
        "MOTORCYCLE": "a motorcycle",
        "BICYCLIST": "a person riding a bicycle",
        "VEHICULAR_TRAILER": "a trailer",
        "TRUCK_CAB": "a semi truck",
        "MOTORCYCLIST": "a person riding a motorcycle", 
        "DOG": "a dog",
        "SCHOOL_BUS": "a yellow school bus",
        "WHEELED_RIDER": "a person on a skateboard",
        "STROLLER": "a baby stroller",
        "ARTICULATED_BUS": "an articulated bus",
        "MESSAGE_BOARD_TRAILER": "an electronic message board",
        "MOBILE_PEDESTRIAN_SIGN": "a pedestrian crossing sign",
        "WHEELCHAIR": "a wheelchair", 
        "RAILED_VEHICLE": "a train",
        "OFFICIAL_SIGNALER": "a traffic controller",
        "TRAFFIC_LIGHT_TRAILER": "a portable traffic light",
        "ANIMAL": "an animal"
    }

# Enhanced implementation using multiple prompts per class
def get_enhanced_text_features(pretrained_name, class_names):
    """
    Generate enhanced text features using multiple prompts per class
    """
    all_prompts = []
    class_to_prompt_indices = {}

    model = Owlv2ForObjectDetection.from_pretrained(pretrained_name)
    processor = AutoProcessor.from_pretrained(pretrained_name)
    
    current_idx = 0
    for class_name in class_names:
        if class_name in ARGOVERSE_PROMPTS:
            prompts = ARGOVERSE_PROMPTS[class_name]
            class_to_prompt_indices[class_name] = list(range(current_idx, current_idx + len(prompts)))
            all_prompts.extend(prompts)
            current_idx += len(prompts)
        else:
            # Fallback for unknown classes
            prompt = f"a {class_name.lower().replace('_', ' ')}"
            class_to_prompt_indices[class_name] = [current_idx]
            all_prompts.append(prompt)
            current_idx += 1
    
    # Get text features for all prompts
    inputs = processor(text=[all_prompts], return_tensors="pt")
    text_features = model.owlv2.get_text_features(**inputs)
    
    # Average features for each class (if using multiple prompts)
    class_features = {}
    for class_name, indices in class_to_prompt_indices.items():
        if len(indices) == 1:
            class_features[class_name] = text_features[indices[0]]
        else:
            # Average multiple prompts for the same class
            class_text_features = text_features[indices]
            class_features[class_name] = class_text_features.mean(dim=0)
    
    return class_features, text_features

class OWLv2Detector(nn.Module):
    def __init__(self, pretrained_name, class_names):
        super().__init__()

        self.pretrained_name = pretrained_name
        
        model = Owlv2ForObjectDetection.from_pretrained(self.pretrained_name)
        print('model.config', model.config)

        self.image_size = model.config.vision_config.image_size
        print('self.image_size', self.image_size)

        self.vision_model = model.owlv2.vision_model

        self.text_features = None

        if len(class_names) > 1: # not placeholder (zero-shot inference)
            processor = AutoProcessor.from_pretrained(self.pretrained_name)
            inputs = processor(
                text=[[f"a {x}" for x in class_names]], return_tensors="pt"
            )
            self.text_features = model.owlv2.get_text_features(**inputs)
            print('self.text_features', self.text_features.shape)

        self.image_processor = Owlv2ImageProcessor.from_pretrained(self.pretrained_name)

        self.class_head = model.class_head
        self.box_head = model.box_head
        self.objectness_head = model.objectness_head
        self.layer_norm = model.layer_norm
        self.sigmoid = nn.Sigmoid()

        self.vision_model.requires_grad_(False)
        self.class_head.requires_grad_(False)
        self.box_head.requires_grad_(False)
        self.layer_norm.requires_grad_(False)

    def preprocess_image(self, image):
        return self.image_processor.preprocess(images=image, return_tensors='pt')

    def preprocess_batch(self, images):
        """
        Preprocess a batch of images for the model.
        
        Args:
            images: List of images to process (numpy arrays in RGB format)
            
        Returns:
            Dictionary containing preprocessed batch with pixel values tensor
        """
        if not isinstance(images, list):
            images = [images]
            
        # Process all images in a single batch
        return self.image_processor.preprocess(images=images, return_tensors='pt')

    # Copied from transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.normalize_grid_corner_coordinates
    def normalize_grid_corner_coordinates(self, feature_map: torch.FloatTensor):
        # Computes normalized xy corner coordinates from feature_map.
        if not feature_map.ndim == 4:
            raise ValueError("Expected input shape is [batch_size, num_patches, num_patches, hidden_dim]")

        device = feature_map.device
        num_patches = feature_map.shape[1]

        # TODO: Remove numpy usage.
        box_coordinates = np.stack(
            np.meshgrid(np.arange(1, num_patches + 1), np.arange(1, num_patches + 1)), axis=-1
        ).astype(np.float32)
        box_coordinates /= np.array([num_patches, num_patches], np.float32)

        # Flatten (h, w, 2) -> (h*w, 2)
        box_coordinates = box_coordinates.reshape(
            box_coordinates.shape[0] * box_coordinates.shape[1], box_coordinates.shape[2]
        )
        box_coordinates = torch.from_numpy(box_coordinates).to(device)

        return box_coordinates

    def objectness_predictor(self, image_features: torch.FloatTensor) -> torch.FloatTensor:
        """Predicts the probability that each image feature token is an object.

        Args:
            image_features (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_dim)`)):
                Features extracted from the image.
        Returns:
            Objectness scores.
        """
        image_features = image_features.detach()
        objectness_logits = self.objectness_head(image_features)
        objectness_logits = objectness_logits[..., 0]
        return objectness_logits

    # Copied from transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.compute_box_bias
    def compute_box_bias(self, feature_map: torch.FloatTensor) -> torch.FloatTensor:
        # The box center is biased to its position on the feature grid
        box_coordinates = self.normalize_grid_corner_coordinates(feature_map)
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # Unnormalize xy
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

        # The box size is biased to the patch size
        box_size = torch.full_like(box_coord_bias, 1.0 / feature_map.shape[-2])
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # Compute box bias
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias

    # Copied from transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.box_predictor
    def box_predictor(
        self,
        image_feats: torch.FloatTensor,
        feature_map: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            image_feats:
                Features extracted from the image, returned by the `image_text_embedder` method.
            feature_map:
                A spatial re-arrangement of image_features, also returned by the `image_text_embedder` method.
        Returns:
            pred_boxes:
                List of predicted boxes (cxcywh normalized to 0, 1) nested within a dictionary.
        """
        # Bounding box detection head [batch_size, num_boxes, 4].
        pred_boxes = self.box_head(image_feats)

        # Compute the location of each token on the grid and use it to compute a bias for the bbox prediction
        pred_boxes += self.compute_box_bias(feature_map)
        pred_boxes = self.sigmoid(pred_boxes)
        return pred_boxes

    # Copied from transformers.models.owlvit.modeling_owlvit.OwlViTForObjectDetection.class_predictor
    def class_predictor(
        self,
        image_feats: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            image_feats:
                Features extracted from the `image_text_embedder`.
            query_embeds:
                Text query embeddings.
            query_mask:
                Must be provided with query_embeddings. A mask indicating which query embeddings are valid.
        """
        (pred_logits, image_class_embeds) = self.class_head(image_feats, query_embeds, query_mask)

        return (pred_logits, image_class_embeds)

    def get_image_class_embeds(self, image_feats: torch.FloatTensor):
        image_class_embeds = self.class_head.dense0(image_feats)

        # Normalize image and text features
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)

        return image_class_embeds

    def post_process_boxes(self, pred_bboxes_xywh, target_size=[768, 768]):
        xy = pred_bboxes_xywh[..., 0:2]
        wh = pred_bboxes_xywh[..., 2:]

        xy1 = xy - wh * 0.5
        xy2 = xy + wh * 0.5
        xyxy = torch.cat((xy1, xy2), dim=-1)

        scale_fct = xyxy.new_tensor([target_size[1], target_size[0], target_size[1], target_size[0]]).reshape(1, 1, 4)
        # scale_fct = xyxy.new_tensor([1600, 900, 1600, 900]).reshape_as(xyxy)

        xyxy_scaled = xyxy * scale_fct

        return xyxy_scaled

    def forward_batch(self, pixel_values, target_size=[900, 1600]):
        """
        Process a batch of images efficiently.
        
        Args:
            pixel_values: Tensor of shape [batch_size, channels, height, width]
            target_size: Target size for the output boxes
            
        Returns:
            Dictionary containing predictions for the entire batch
        """
        batch_size = pixel_values.shape[0]
        
        # Move inputs to the same device as the model
        device = next(self.parameters()).device
        pixel_values = pixel_values.to(device)
        
        with torch.no_grad():  # Disable gradient calculation for inference
            # Get vision model outputs
            last_hidden_state, pooled_output = self.vision_model(pixel_values, return_dict=False)
            
            # Process hidden states
            image_embeds = self.vision_model.post_layernorm(last_hidden_state)
            
            # Resize class token for the whole batch at once
            new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
            class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)
            
            # Merge image embedding with class tokens
            image_embeds = image_embeds[:, 1:, :] * class_token_out
            image_embeds = self.layer_norm(image_embeds)
            
            # Reshape to [batch_size, num_patches, num_patches, hidden_size]
            patch_size = int(np.sqrt(image_embeds.shape[1]))
            new_size = (batch_size, patch_size, patch_size, image_embeds.shape[-1])
            feature_map = image_embeds.reshape(new_size)
            
            # Reshape for further processing
            image_feats = feature_map.reshape(batch_size, patch_size * patch_size, feature_map.shape[-1])
            
            # Run all predictions in parallel
            pred_boxes = self.box_predictor(image_feats, feature_map)
            pred_boxes = self.post_process_boxes(pred_boxes, target_size)
            
            objectness_logits = self.objectness_predictor(image_feats)
            image_class_embeds = self.get_image_class_embeds(image_feats)

            print('image_class_embeds, image_feats', image_class_embeds.shape, image_feats.shape)
            
            # Return all outputs in a dictionary
            return {
                'pred_boxes': pred_boxes, 
                'image_feats': image_feats, 
                'objectness_logits': objectness_logits, 
                'image_class_embeds': image_class_embeds
            }

    def forward(self, images, target_size=[900, 1600]):
        # Embed images
        last_hidden_state, pooled_output = self.vision_model(images, return_dict=False)

        image_embeds = self.vision_model.post_layernorm(last_hidden_state)

        # Resize class token
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)

        # changed name from image_text_embedder
        feature_map = image_embeds

        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))

        # Reshape from [batch_size * max_text_queries, hidden_dim] -> [batch_size, max_text_queries, hidden_dim]
        # max_text_queries = input_ids.shape[0] // batch_size
        # query_embeds = query_embeds.reshape(batch_size, max_text_queries, query_embeds.shape[-1])

        # If first token is 0, then this is a padded query [batch_size, num_queries].
        # input_ids = input_ids.reshape(batch_size, max_text_queries, input_ids.shape[-1])
        # query_mask = input_ids[..., 0] > 0

        # Predict object classes [batch_size, num_patches, num_queries+1]
        # (pred_logits, class_embeds) = self.class_predictor(image_feats, query_embeds, query_mask)

        # Predict object boxes
        pred_boxes = self.box_predictor(image_feats, feature_map)
        pred_boxes = self.post_process_boxes(pred_boxes, target_size)
        
        # Predict objectness
        objectness_logits = self.objectness_predictor(image_feats)

        # image class embeddings
        image_class_embeds = self.get_image_class_embeds(image_feats)

        # return pred_boxes, image_feats, objectness_logits
        return dict(pred_boxes=pred_boxes, image_feats=image_feats, objectness_logits=objectness_logits, image_class_embeds=image_class_embeds)


def get_ring_camera_paths(log_dir: Path) -> Dict[str, Path]:
    """Get paths to all ring camera directories for a log."""
    cameras_dir = log_dir / "sensors" / "cameras"
    ring_cameras = {}
    
    for camera_dir in cameras_dir.iterdir():
        if camera_dir.name.startswith("ring_"):
            ring_cameras[camera_dir.name] = camera_dir
    
    return ring_cameras

def get_image_files(camera_dir: Path) -> List[tuple]:
    """Get all image files with their timestamps."""
    image_files = []
    for img_file in camera_dir.glob("*.jpg"):
        timestamp_ns = int(img_file.stem)
        image_files.append((timestamp_ns, img_file))
    
    # Sort by timestamp
    image_files.sort(key=lambda x: x[0])
    return image_files

    
def convert_to_original_coords(boxes, original_shape=(1920, 1080), resized_shape=(1008, 1008)):
    """
    Convert bounding box coordinates from resized/padded image to original image size.

    Parameters:
    - boxes: Nx4 NumPy array of bounding boxes in the resized image [x_min, y_min, x_max, y_max].
    - original_shape: Tuple of (width, height) for the original image.
    - resized_shape: Tuple of (width, height) for the resized image.

    Returns:
    - original_boxes: Nx4 NumPy array of bounding boxes in the original image size.
    """
    original_shape = np.array(original_shape)
    resized_shape = np.array(resized_shape)

    ldim = np.argmax(original_shape)

    rev_scale_f = resized_shape[ldim] / original_shape[ldim]

    pad = resized_shape - original_shape * rev_scale_f  

    scale_w = (original_shape[0]) / (resized_shape[0] - pad[0])
    scale_h = (original_shape[1]) / (resized_shape[1] - pad[1])

    # Adjust coordinates
    boxes[:, [0, 2]] = (boxes[:, [0, 2]]) * scale_w
    boxes[:, [1, 3]] = (boxes[:, [1, 3]]) * scale_h


    # Clip coordinates to stay within original image boundaries
    boxes[:, 0] = np.clip(boxes[:, 0], 0, original_shape[0])
    boxes[:, 1] = np.clip(boxes[:, 1], 0, original_shape[1])
    boxes[:, 2] = np.clip(boxes[:, 2], 0, original_shape[0])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, original_shape[1])

    return boxes

def process_argoverse_log(log_dir: Path, owlvit_model, output_dir: Path, 
                         batch_size: int = 4, top_k: int = 100, 
                         conf_thresh: float = 0.1, vis: bool = False):
    """
    Process a single Argoverse log efficiently.
    
    Args:
        log_dir: Path to the log directory
        owlvit_model: OWLv2Detector model instance
        output_dir: Directory to save predictions
        batch_size: Number of images to process in a batch
        top_k: Number of top boxes to keep per image
        conf_thresh: Confidence threshold for filtering
        vis: Whether to create visualizations (only for first timestamp)
    """
    log_id = log_dir.name
    log_output_dir = output_dir / log_id
    log_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all ring cameras
    ring_cameras = get_ring_camera_paths(log_dir)
    print(f"Found {len(ring_cameras)} ring cameras: {list(ring_cameras.keys())}")
    
    # Group images by timestamp across all cameras
    timestamp_to_images = {}

    skipped = 0
    todo = 0
    
    for camera_name, camera_dir in ring_cameras.items():
        image_files = get_image_files(camera_dir)
        print(f"{camera_name}: {len(image_files)} images")
        
        for timestamp_ns, img_path in image_files:
            if timestamp_ns not in timestamp_to_images:
                timestamp_to_images[timestamp_ns] = {}

            if not (log_output_dir / f"{timestamp_ns}_{camera_name}.pkl").exists():
                timestamp_to_images[timestamp_ns][camera_name] = img_path
                todo += 1
            else:
                skipped += 1

    # print('skipped', skipped)
    # exit()

    if todo == 0:
        print(f'done all for {log_id=}')
        return

    # Sort timestamps
    sorted_timestamps = sorted(timestamp_to_images.keys())
    print(f"Processing {len(sorted_timestamps)} timestamps")
    
    # Process images in batches across cameras and timestamps
    first_timestamp_processed = {camera_name: False for camera_name in ring_cameras.keys()}
    
    for ts_idx, timestamp_ns in enumerate(sorted_timestamps):
        print(f"Processing timestamp {ts_idx+1}/{len(sorted_timestamps)}: {timestamp_ns}")
        
        # Check if all cameras have images for this timestamp
        # cameras_with_images = timestamp_to_images[timestamp_ns]
        # if len(cameras_with_images) != len(ring_cameras):
        #     print(f"  Skipping timestamp {timestamp_ns} - missing cameras")
        #     print(f"cameras_with_images={len(cameras_with_images)} ring_cameras={len(ring_cameras)}")
        #     print(f"{cameras_with_images=} {ring_cameras=}")
        #     exit()
        #     continue
        
        cameras_with_images = timestamp_to_images[timestamp_ns]
        if len(cameras_with_images) == 0:
            print(f"  Skipping timestamp {timestamp_ns} - no cameras")
            continue
        
        print(f"  Processing {len(cameras_with_images)}/{len(ring_cameras)} cameras for timestamp {timestamp_ns}")

        # Load all images for this timestamp
        batch_images = []
        image_metas = []
        
        for camera_name in sorted(cameras_with_images.keys()):
            img_path = cameras_with_images[camera_name]
            
            # Load image
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            batch_images.append(img_rgb)
            image_metas.append({
                'camera_name': camera_name,
                'timestamp_ns': timestamp_ns,
                'image_path': img_path,
                'image_bgr': img,
                'shape': [img_rgb.shape[1], img_rgb.shape[0]]  # width, height
            })
        
        # Process in smaller batches if needed
        for batch_start in range(0, len(batch_images), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_images))
            current_batch = batch_images[batch_start:batch_end]
            current_metas = image_metas[batch_start:batch_end]
            
            # Preprocess batch
            owlvit_data = owlvit_model.preprocess_batch(current_batch)
            
            # Run inference
            with torch.no_grad():
                vit_outputs = owlvit_model.forward_batch(
                    owlvit_data['pixel_values'].to(next(owlvit_model.parameters()).device),
                    target_size=[owlvit_model.image_size, owlvit_model.image_size]
                )
            
            # Process results for each image in the batch
            for i, meta in enumerate(current_metas):
                # Extract outputs for this image
                pred_boxes = vit_outputs['pred_boxes'][i].detach().cpu().numpy()
                objectness_scores = vit_outputs['objectness_logits'][i].sigmoid().detach().cpu().numpy()
                image_class_embeds = vit_outputs['image_class_embeds'][i].detach().cpu().numpy()
                image_feats = vit_outputs['image_feats'][i].detach().cpu().numpy()

                pred_boxes = convert_to_original_coords(
                    pred_boxes.reshape(-1, 4),
                    meta['shape'],
                    [owlvit_model.image_size, owlvit_model.image_size]
                )

                # Compute width/height
                widths = pred_boxes[:, 2] - pred_boxes[:, 0]
                heights = pred_boxes[:, 3] - pred_boxes[:, 1]

                # Filter out small boxes (< 32 pixels in width or height)
                size_mask = (widths >= 32) & (heights >= 32)

                pred_boxes = pred_boxes[size_mask]
                objectness_scores = objectness_scores[size_mask]
                image_class_embeds = image_class_embeds[size_mask]
                image_feats = image_feats[size_mask]
                
                # Apply confidence filtering
                conf_mask = objectness_scores > conf_thresh
                filtered_boxes = pred_boxes[conf_mask]
                filtered_logits = objectness_scores[conf_mask]
                filtered_embeds = image_class_embeds[conf_mask]
                filtered_feats = image_feats[conf_mask]

                # If fewer than top_k remain, fall back to top_k overall
                # if len(filtered_logits) < top_k:
                #     top_indices = np.argsort(objectness_scores)[-top_k:]
                #     filtered_boxes = pred_boxes[top_indices]
                #     filtered_logits = objectness_scores[top_indices]
                #     filtered_embeds = image_class_embeds[top_indices]
                #     filtered_feats = image_feats[top_indices]
                # else:
                if len(filtered_logits) > top_k:
                    # Otherwise, keep only the top_k from the filtered set
                    top_indices = np.argsort(filtered_logits)[-top_k:]
                    filtered_boxes = filtered_boxes[top_indices]
                    filtered_logits = filtered_logits[top_indices]
                    filtered_embeds = filtered_embeds[top_indices]
                    filtered_feats = filtered_feats[top_indices]
                
                # Prepare output data
                output_data = {
                    'pred_boxes': filtered_boxes,
                    'objectness_scores': filtered_logits,
                    'image_class_embeds': filtered_embeds,
                    'image_feats': filtered_feats,
                    'metadata': {
                        'timestamp_ns': meta['timestamp_ns'],
                        'camera_name': meta['camera_name'],
                        'image_shape': meta['shape'],
                        'num_detections': len(filtered_logits),
                        'conf_thresh': conf_thresh,
                        'top_k': top_k
                    }
                }
                
                # Save predictions
                save_path = log_output_dir / f"{meta['timestamp_ns']}_{meta['camera_name']}.pkl"
                with open(save_path, 'wb') as f:
                    pickle.dump(output_data, f)
                
                # Print file size
                file_size = save_path.stat().st_size / (1024 * 1024)  # MB
                print(f"  Saved {meta['camera_name']}: {len(filtered_logits)} boxes, {file_size:.2f}MB")
                
                # Create visualization for first timestamp only
                if vis and not first_timestamp_processed[meta['camera_name']]:
                    create_visualization(meta, output_data, log_output_dir)
                    first_timestamp_processed[meta['camera_name']] = True
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def create_visualization(meta: Dict[str, Any], output_data: Dict[str, Any], 
                        output_dir: Path):
    """Create visualization of detections."""
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Load image
    img_path = meta['image_path']
    image = cv2.imread(str(img_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title(f"{meta['camera_name']} - {meta['timestamp_ns']}\n"
                f"{len(output_data['pred_boxes'])} detections")
    
    # Draw bounding boxes
    boxes = output_data['pred_boxes']
    scores = output_data['objectness_scores']
    
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle
        rect = patches.Rectangle((x1, y1), width, height,
                               linewidth=2, edgecolor='red', 
                               facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Add score text
        ax.text(x1, y1-5, f'{score:.3f}', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
               fontsize=8, color='white')
    
    ax.axis('off')
    
    # Save visualization
    vis_path = vis_dir / f"{meta['timestamp_ns']}_{meta['camera_name']}_detections.png"
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization: {vis_path}")

def load_predictions(output_dir: Path, log_id: str, timestamp_ns: int, 
                    camera_name: str) -> Dict[str, Any]:
    """
    Load predictions for a specific log, timestamp, and camera.
    
    Args:
        output_dir: Base output directory
        log_id: Log identifier
        timestamp_ns: Timestamp in nanoseconds
        camera_name: Camera name (e.g., 'ring_front_center')
    
    Returns:
        Dictionary containing predictions and metadata
    """
    pred_path = output_dir / log_id / f"{timestamp_ns}_{camera_name}.pkl"
    
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")
    
    with open(pred_path, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description='Process Argoverse logs with OWLv2')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to Argoverse data root directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save predictions')
    parser.add_argument('--log_ids', type=str, nargs='+',
                       help='Specific log IDs to process (if not provided, processes all)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for processing')
    parser.add_argument('--top_k', type=int, default=100,
                       help='Number of top boxes to keep per image')
    parser.add_argument('--conf_thresh', type=float, default=0.1,
                       help='Confidence threshold for filtering')
    parser.add_argument('--vis', action='store_true',
                       help='Create visualizations for first log and timestamp')
    
    args = parser.parse_args()


    class_features, text_features = get_enhanced_text_features('google/owlv2-large-patch14-ensemble', ARGOVERSE_PROMPTS.keys())
    
    # print("class_features", class_features.shape)
    print("text_features", text_features.shape)

    torch.save(class_features, './class_features.pt')
    torch.save(text_features, './text_features.pt')

    exit()


    # Initialize your OWLv2 model here
    pretrained_name = 'google/owlv2-large-patch14-ensemble'
    owlvit_model = OWLv2Detector(pretrained_name, ['placeholder']).to('cuda:1')
    owlvit_model.eval()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get log directories to process
    if args.log_ids:
        log_dirs = [data_root / log_id for log_id in args.log_ids]
    else:
        log_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    
    print(f"Processing {len(log_dirs)} logs...")
    
    for i, log_dir in enumerate(log_dirs):
        if not log_dir.exists():
            print(f"Log directory not found: {log_dir}")
            continue
            
        print(f"\n[{i+1}/{len(log_dirs)}] Processing log: {log_dir.name}")
        
        # Only visualize for the first log if vis is enabled
        vis_enabled = args.vis and i == 0
        
        try:
            process_argoverse_log(
                log_dir=log_dir,
                owlvit_model=owlvit_model,  # You need to initialize this
                output_dir=output_dir,
                batch_size=args.batch_size,
                top_k=args.top_k,
                conf_thresh=args.conf_thresh,
                vis=vis_enabled
            )
        except Exception as e:
            print(f"Error processing {log_dir.name}: {e}")
            continue
    
    print("\nProcessing complete!")
    
    # Example of how to load predictions
    if log_dirs:
        example_log = log_dirs[0].name
        print(f"\nExample: Loading predictions for log {example_log}")
        print("Use: load_predictions(output_dir, log_id, timestamp_ns, camera_name)")

if __name__ == "__main__":
    main()