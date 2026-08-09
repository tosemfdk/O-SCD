"""
Interactive Gaussian Splatting Viewer using Viser + CUDA Rasterization.

Renders photorealistic Gaussian splats using the exact CUDA rasterizer
(diff-gaussian-rasterization) and displays them interactively in a Viser
web interface. Supports toggling between Reference, Updated, and Change 3DGS scenes.

Usage:
    conda activate scene
    python viser_viewer.py \
        --ref_ply <path_to_reference_point_cloud.ply> \
        --updated_ply <path_to_updated_scene.ply> \
        --change_ply <path_to_change_scene.ply> \
        --inference_dir <path_to_inference_images> \
        --port 8080
"""

import sys
import os
import json
import math
import time
import threading
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import cv2
import numpy as np
import torch
import viser

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from gaussian_renderer import render, render_change
from temporal.viewer_export import ensure_dc_state_viewer_export
from utils.graphics_utils import getProjectionMatrix


@dataclass
class PipelineConfig:
    """Mimics PipelineParams for the renderer."""
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False
    debug: bool = False
    antialiasing: bool = False


def quaternion_to_rotation_matrix(wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def colmap_camera_to_viser(position: np.ndarray, rotation: np.ndarray, fx: float, fy: float, width: int, height: int, resolution: int = 1) -> dict:
    """
    Convert a COLMAP/3DGS camera (from cameras.json) to Viser camera parameters.
    """
    R_c2w_colmap = np.array(rotation, dtype=np.float32)
    
    flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R_c2w_opengl = R_c2w_colmap @ flip
    
    wxyz = rotation_matrix_to_quaternion(R_c2w_opengl)
    
    train_w = width // resolution
    train_h = height // resolution
    
    fy_scaled = fy / resolution
    fovy = 2.0 * math.atan(train_h / (2.0 * fy_scaled))
    
    return {
        "wxyz": wxyz,
        "position": np.array(position, dtype=np.float32),
        "fov": fovy,
        "width": train_w,
        "height": train_h,
    }


def load_cameras_json(path: str) -> List[Dict]:
    with open(path, "r") as f:
        cameras = json.load(f)
    print(f"  → {len(cameras)} camera viewpoints loaded")
    return cameras


def create_mini_cam(
    wxyz: np.ndarray,
    position: np.ndarray,
    fov: float,
    aspect: float,
    width: int,
    height: int,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> MiniCam:
    R_c2w_opengl = quaternion_to_rotation_matrix(wxyz)
    
    flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R_c2w_colmap = R_c2w_opengl @ flip
    
    R_w2c = R_c2w_colmap.T
    t = -R_w2c @ position
    
    W2C = np.eye(4, dtype=np.float32)
    W2C[:3, :3] = R_w2c
    W2C[:3, 3] = t
    
    FoVy = fov
    FoVx = 2.0 * math.atan(math.tan(FoVy / 2.0) * aspect)
    
    proj = getProjectionMatrix(znear, zfar, FoVx, FoVy).transpose(0, 1).cuda()
    
    world_view = torch.tensor(W2C, dtype=torch.float32).transpose(0, 1).cuda()
    
    full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    
    return MiniCam(
        width=width,
        height=height,
        fovy=FoVy,
        fovx=FoVx,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view,
        full_proj_transform=full_proj,
    )


class GaussianViewer:
    def __init__(
        self,
        ref_ply: str,
        updated_ply: Optional[str] = None,
        change_ply: Optional[str] = None,
        cameras_json: Optional[str] = None,
        inference_dir: Optional[str] = None,
        sh_degree: int = 3,
        port: int = 8080,
        resolution: int = 4,
        initial_scene: str = "reference",
        change_display: str = "binary",
    ):
        self.port = port
        self.sh_degree = sh_degree
        self.resolution = resolution
        self.pipe = PipelineConfig()
        self.background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        
        self.viewpoints = []
        self.viewpoint_names = []
        self.inference_images = {}
        self.inference_thumbnails = {}
        self._current_viewpoint_idx = 0
        
        self.render_width = 1008
        self.active_scene = initial_scene
        self.change_display = change_display
        self._render_lock = threading.Lock()
        
        print(f"Loading reference model from: {ref_ply}")
        self.gaussians_ref = GaussianModel(sh_degree, sh_degree)
        self.gaussians_ref.load_ply(ref_ply)
        print(f"  → {self.gaussians_ref.get_xyz.shape[0]:,} Gaussians loaded")
        
        self.gaussians_upd = None
        if updated_ply and os.path.exists(updated_ply):
            try:
                print(f"Loading updated model from: {updated_ply}")
                self.gaussians_upd = GaussianModel(sh_degree, sh_degree)
                self.gaussians_upd.load_ply(updated_ply)
                print(f"  → {self.gaussians_upd.get_xyz.shape[0]:,} Gaussians loaded")
            except Exception as e:
                print(f"Error loading updated model from {updated_ply}: {e}")
                self.gaussians_upd = None
        else:
            if updated_ply:
                print(f"Warning: Updated PLY not found at {updated_ply}")

        self.gaussians_change = None
        if change_ply and os.path.exists(change_ply):
            try:
                print(f"Loading change model from: {change_ply}")
                self.gaussians_change = GaussianModel(sh_degree, 0)
                self.gaussians_change.load_ply(change_ply)
                print(f"  → {self.gaussians_change.get_xyz.shape[0]:,} Gaussians loaded")
            except Exception as e:
                print(f"Error loading change model from {change_ply}: {e}")
                self.gaussians_change = None
        else:
            if change_ply:
                print(f"Warning: Change PLY not found at {change_ply}")

        if self.active_scene == "change" and self.gaussians_change is None:
            self.active_scene = "reference"
        elif self.active_scene == "updated" and self.gaussians_upd is None:
            self.active_scene = "reference"

        cameras_path = cameras_json
        if not cameras_path or not os.path.exists(cameras_path):
            ref_dir = os.path.dirname(os.path.dirname(os.path.dirname(ref_ply)))
            auto_path = os.path.join(ref_dir, "cameras.json")
            if os.path.exists(auto_path):
                cameras_path = auto_path
                print(f"Auto-detected cameras.json at: {cameras_path}")
        
        if cameras_path and os.path.exists(cameras_path):
            raw_cameras = load_cameras_json(cameras_path)
            for cam in raw_cameras:
                viser_cam = colmap_camera_to_viser(
                    cam["position"], cam["rotation"],
                    cam["fx"], cam["fy"], cam["width"], cam["height"],
                    resolution=self.resolution,
                )
                self.viewpoints.append(viser_cam)
                self.viewpoint_names.append(cam["img_name"])
            if self.viewpoints:
                self.render_width = 2048 # Force default 2048 instead of `self.viewpoints[0]["width"]`
        else:
            self.render_width = 2048
            print("No cameras.json found. Free-flight mode only.")
        
        # First load the inference images, and filter the viewpoints simultaneously
        self._load_inference_images(inference_dir, ref_ply)
        
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self._setup_gui()
        self._setup_callbacks()
        
        print(f"\n{'='*60}")
        print(f"  Gaussian Splatting Viewer running at:")
        print(f"  http://localhost:{port}")
        if self.viewpoints:
            print(f"  {len(self.viewpoints)} camera viewpoints available")
            print(f"  Training resolution: {self.viewpoints[0]['width']}×{self.viewpoints[0]['height']} (1/{self.resolution})")
        if self.inference_images:
            print(f"  {len(self.inference_images)} inference images loaded")
        print(f"{'='*60}\n")
    
    def _load_inference_images(self, inference_dir: Optional[str], ref_ply: str):
        if inference_dir and os.path.isdir(inference_dir):
            img_dir = inference_dir
        else:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(ref_ply))))
            img_dir = os.path.join(base, "inference_scene", "images")
            if not os.path.isdir(img_dir):
                img_dir = os.path.join(base, "reference_scene", "images")
        
        if not os.path.isdir(img_dir):
            print(f"No inference images directory found at: {img_dir}")
            return
        
        print(f"Loading inference images from: {img_dir}")
        loaded = 0
        
        # Get all files in inference directory
        available_inference_files = []
        if os.path.isdir(img_dir):
            available_inference_files = os.listdir(img_dir)
            
        filtered_viewpoints = []
        filtered_names = []
        
        for idx, name in enumerate(self.viewpoint_names):
            # name is typically something like "Inst_1_IMG_E7280"
            # Extract the core identifier, e.g., "E7280"
            parts = name.split('_')
            core_id = parts[-1] if parts else name
            
            # Find a matching file in the inference directory
            matching_file = None
            for f in available_inference_files:
                if core_id in f:
                    matching_file = f
                    break
                    
            if matching_file:
                path = os.path.join(img_dir, matching_file)
                img = cv2.imread(path)
                if img is not None:
                    filtered_viewpoints.append(self.viewpoints[idx])
                    filtered_names.append(name)
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if self.viewpoints:
                        h = self.viewpoints[0]["height"]
                        w = self.viewpoints[0]["width"]
                        img_full = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    else:
                        img_full = img
                    self.inference_images[name] = img_full
                    
                    thumb_w = 260
                    thumb_h = int(img_full.shape[0] * thumb_w / img_full.shape[1])
                    thumb_img = cv2.resize(img_full, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

                    # Only store the base inference image here; we will composite on the fly
                    self.inference_thumbnails[name] = thumb_img
                    loaded += 1
        
        # Override the viewpoints with only those that have inference images available
        if loaded > 0:
            self.viewpoints = filtered_viewpoints
            self.viewpoint_names = filtered_names

        print(f"  → {loaded} inference images loaded (Filtered viewpoints accordingly)")
    
    def _get_active_gaussians(self) -> GaussianModel:
        if self.active_scene == "updated" and self.gaussians_upd is not None:
            return self.gaussians_upd
        if self.active_scene == "change" and self.gaussians_change is not None:
            return self.gaussians_change
        return self.gaussians_ref
    
    def _setup_gui(self):
        with self.server.gui.add_folder("Scene"):
            scene_options = ["Reference 3DGS"]
            if self.gaussians_upd is not None:
                scene_options.append("Updated 3DGS")
            if self.gaussians_change is not None:
                scene_options.append("Change 3DGS")
            
            self.gui_scene = self.server.gui.add_dropdown(
                "Active Scene",
                options=scene_options,
                initial_value={
                    "reference": "Reference 3DGS",
                    "updated": "Updated 3DGS",
                    "change": "Change 3DGS",
                }[self.active_scene],
            )

            self.gui_change_display = self.server.gui.add_dropdown(
                "R_change Display",
                options=["Raw Response", "Binary Mask"],
                initial_value="Raw Response" if self.change_display == "raw" else "Binary Mask",
                disabled=self.gaussians_change is None,
            )

            self.gui_change_threshold = self.server.gui.add_slider(
                "R_change Threshold",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=0.5,
                disabled=self.gaussians_change is None,
            )

            # Add Overlay Checkbox
            self.gui_overlay = self.server.gui.add_checkbox(
                "Overlay Change Mask",
                initial_value=False,
                disabled=self.gaussians_change is None
            )

            self.gui_overlay_opacity = self.server.gui.add_slider(
                "Overlay Opacity",
                min=0.1,
                max=1.0,
                step=0.1,
                initial_value=0.5,
                disabled=self.gaussians_change is None
            )

            self.gui_overlay_color = self.server.gui.add_dropdown(
                "Overlay Color",
                options=["White", "Red", "Green", "Blue", "Yellow"],
                initial_value="White",
                disabled=self.gaussians_change is None
            )

        if self.viewpoints:
            with self.server.gui.add_folder("Camera Viewpoints"):
                self.gui_viewpoint = self.server.gui.add_dropdown(
                    "Viewpoint",
                    options=self.viewpoint_names,
                    initial_value=self.viewpoint_names[0],
                )
                with self.server.gui.add_folder("Navigation", expand_by_default=True):
                    self.gui_prev_btn = self.server.gui.add_button("◀ Previous")
                    self.gui_next_btn = self.server.gui.add_button("Next ▶")
                    self.gui_viewpoint_info = self.server.gui.add_markdown(
                        f"**Viewpoint 1 / {len(self.viewpoints)}**"
                    )
                
                if self.inference_thumbnails:
                    first_name = self.viewpoint_names[0]
                    if first_name in self.inference_thumbnails:
                        base_img = self.inference_thumbnails[first_name]
                        # Create dummy composite 3x height of base thumbnail
                        dummy_composite = np.zeros((base_img.shape[0] * 3, base_img.shape[1], 3), dtype=np.uint8)
                        self.gui_inference_image = self.server.gui.add_image(
                            dummy_composite,
                            label="Dynamic Preview (Rendered)",
                        )
        
        with self.server.gui.add_folder("Rendering"):
            slider_min = 256    
            slider_max = 2048
            slider_val = max(slider_min, min(self.render_width, slider_max))
            
            self.gui_render_width = self.server.gui.add_slider(
                "Render Width",
                min=slider_min,
                max=slider_max,
                step=64,
                initial_value=slider_val,
            )
    
    def _setup_callbacks(self):
        @self.gui_scene.on_update
        def _on_scene_change(event: viser.GuiEvent) -> None:
            if event.client is None:
                return
            if self.gui_scene.value == "Updated 3DGS":
                self.active_scene = "updated"
            elif self.gui_scene.value == "Change 3DGS":
                self.active_scene = "change"
            else:
                self.active_scene = "reference"
            self._render_for_client(event.client)
        
        @self.gui_overlay.on_update
        def _on_overlay_change(event: viser.GuiEvent) -> None:
            if event.client is None:
                return
            self._render_for_client(event.client)

        @self.gui_overlay_opacity.on_update
        def _on_opacity_change(event: viser.GuiEvent) -> None:
            if event.client is None:
                return
            self._render_for_client(event.client)

        @self.gui_overlay_color.on_update
        def _on_color_change(event: viser.GuiEvent) -> None:
            if event.client is None:
                return
            self._render_for_client(event.client)

        @self.gui_change_display.on_update
        def _on_change_display(event: viser.GuiEvent) -> None:
            if event.client is None:
                return
            self.change_display = "raw" if self.gui_change_display.value == "Raw Response" else "binary"
            self._render_for_client(event.client)

        @self.gui_change_threshold.on_update
        def _on_change_threshold(event: viser.GuiEvent) -> None:
            if event.client is None:
                return
            self._render_for_client(event.client)

        @self.gui_render_width.on_update
        def _on_width_change(event: viser.GuiEvent) -> None:
            if event.client is None:
                return
            self.render_width = self.gui_render_width.value
            self._render_for_client(event.client)
        
        # Removed Background Color callback
        
        if self.viewpoints:
            @self.gui_viewpoint.on_update
            def _on_viewpoint_change(event: viser.GuiEvent) -> None:
                if event.client is None:
                    return
                idx = self.viewpoint_names.index(self.gui_viewpoint.value)
                self._current_viewpoint_idx = idx
                self._set_camera_viewpoint(event.client, idx)
            
            @self.gui_prev_btn.on_click
            def _on_prev(event: viser.GuiEvent) -> None:
                if event.client is None:
                    return
                idx = (self._current_viewpoint_idx - 1) % len(self.viewpoints)
                self._current_viewpoint_idx = idx
                self.gui_viewpoint.value = self.viewpoint_names[idx]
                self._set_camera_viewpoint(event.client, idx)
            
            @self.gui_next_btn.on_click
            def _on_next(event: viser.GuiEvent) -> None:
                if event.client is None:
                    return
                idx = (self._current_viewpoint_idx + 1) % len(self.viewpoints)
                self._current_viewpoint_idx = idx
                self.gui_viewpoint.value = self.viewpoint_names[idx]
                self._set_camera_viewpoint(event.client, idx)
        
        # Removed checkbox logic for inference panel
        
        @self.server.on_client_connect
        def _on_connect(client: viser.ClientHandle) -> None:
            if self.viewpoints:
                self._set_camera_viewpoint(client, 0)
            
            @client.camera.on_update
            def _on_camera_update(camera: viser.CameraHandle) -> None:
                self._render_for_client(client)
    
    def _set_camera_viewpoint(self, client: viser.ClientHandle, idx: int):
        vp = self.viewpoints[idx]
        client.camera.wxyz = vp["wxyz"]
        client.camera.position = vp["position"]
        client.camera.fov = vp["fov"]
        if hasattr(self, 'gui_viewpoint_info'):
            self.gui_viewpoint_info.content = f"**Viewpoint {idx + 1} / {len(self.viewpoints)}**"
        self._update_inference_thumbnail(idx)
    
    def _update_inference_thumbnail(self, idx: int):
        pass # Now handled dynamically in _render_for_client

    @staticmethod
    def _raw_change_response(rendered: torch.Tensor) -> torch.Tensor:
        return rendered.mean(dim=0, keepdim=True).clamp(0.0, 1.0)

    def _display_change_response(self, rendered: torch.Tensor, *, force_binary: bool = False) -> torch.Tensor:
        response = self._raw_change_response(rendered)
        if force_binary or self.change_display == "binary":
            response = (response > float(self.gui_change_threshold.value)).float()
        return response.repeat(3, 1, 1)

    @torch.no_grad()
    def _render_for_client(self, client: viser.ClientHandle):
        if not self._render_lock.acquire(blocking=False):
            return
        
        try:
            camera = client.camera
            
            width = self.render_width
            aspect = camera.aspect
            if aspect <= 0:
                aspect = 16.0 / 9.0
            height = int(width / aspect)
            
            height = max(height, 128)
            width = max(width, 128)
            aspect = width / height
            
            fov = camera.fov
            if fov <= 0:
                fov = math.radians(60)
            
            wxyz = np.array(camera.wxyz, dtype=np.float32)
            position = np.array(camera.position, dtype=np.float32)
            
            cam = create_mini_cam(wxyz, position, fov, aspect, width, height)
            
            gaussians = self._get_active_gaussians()
            change_render = None
            
            if self.active_scene == "change":
                result = render_change(cam, gaussians, self.pipe, self.background)
                change_render = result["render"]
                rendered = self._display_change_response(change_render)

            else:
                result = render(cam, gaussians, self.pipe, self.background)
                rendered = result["render"]

                if self.gui_overlay.value and self.gaussians_change is not None:
                    change_render = render_change(cam, self.gaussians_change, self.pipe, self.background)["render"]
                    # Process overlay mask
                    mask_1ch = change_render.mean(dim=0, keepdim=True)
                    mask_binary = (mask_1ch > float(self.gui_change_threshold.value)).float()
                    
                    # Determine overlay color
                    color_map = {
                        "White": [1.0, 1.0, 1.0],
                        "Red": [1.0, 0.0, 0.0],
                        "Green": [0.0, 1.0, 0.0],
                        "Blue": [0.0, 0.0, 1.0],
                        "Yellow": [1.0, 1.0, 0.0],
                    }
                    chosen_color = color_map.get(self.gui_overlay_color.value, [1.0, 1.0, 1.0])
                    overlay_rgb = torch.tensor(chosen_color, device="cuda", dtype=torch.float32).view(3, 1, 1)
                    
                    # Expand mask to RGB with the chosen color
                    colored_overlay = mask_binary * overlay_rgb

                    # Blend
                    alpha = self.gui_overlay_opacity.value
                    rendered = rendered * (1 - alpha * mask_binary) + colored_overlay * alpha

            image = rendered.clamp(0.0, 1.0)
            image_hwc = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            client.scene.set_background_image(image_hwc, format="png")
            
            # Dynamic Composite Rendering for thumbnail
            if hasattr(self, 'gui_inference_image') and self.inference_thumbnails:
                thumb_w = 480
                thumb_h = int(image.shape[1] * thumb_w / image.shape[2])
                
                # Create dynamic thumbnails to stack vertically
                thumb_cam = create_mini_cam(wxyz, position, fov, aspect, thumb_w, thumb_h)
                render_ref_thumb = None
                render_change_thumb = None
                render_change_mask_thumb = None
                render_upd_thumb = None
                
                if getattr(self, 'gaussians_ref', None) is not None:
                    res_ref = render(thumb_cam, self.gaussians_ref, self.pipe, self.background)["render"]
                    res_ref = res_ref.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255
                    render_ref_thumb = np.ascontiguousarray(res_ref.astype(np.uint8))
                    
                if getattr(self, 'gaussians_change', None) is not None:
                    res_chg = render_change(thumb_cam, self.gaussians_change, self.pipe, self.background)["render"]
                    raw_response = self._raw_change_response(res_chg)
                    binary_mask = (
                        raw_response > float(self.gui_change_threshold.value)
                    ).float()
                    raw_response = raw_response.repeat(3, 1, 1)
                    binary_mask = binary_mask.repeat(3, 1, 1)
                    raw_response = raw_response.permute(1, 2, 0).cpu().numpy() * 255
                    binary_mask = binary_mask.permute(1, 2, 0).cpu().numpy() * 255
                    render_change_thumb = np.ascontiguousarray(raw_response.astype(np.uint8))
                    render_change_mask_thumb = np.ascontiguousarray(binary_mask.astype(np.uint8))

                if getattr(self, 'gaussians_upd', None) is not None:
                    res_upd = render(thumb_cam, self.gaussians_upd, self.pipe, self.background)["render"]
                    res_upd = res_upd.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255
                    render_upd_thumb = np.ascontiguousarray(res_upd.astype(np.uint8))

                panels = []
                if render_ref_thumb is not None:
                    # Enforce exact shape
                    if render_ref_thumb.shape[0] != thumb_h or render_ref_thumb.shape[1] != thumb_w:
                        render_ref_thumb = cv2.resize(render_ref_thumb, (thumb_w, thumb_h))
                    cv2.putText(render_ref_thumb, "Reference Scene", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    panels.append(render_ref_thumb)
                    
                if render_change_thumb is not None:
                    # Enforce exact shape
                    if render_change_thumb.shape[0] != thumb_h or render_change_thumb.shape[1] != thumb_w:
                        render_change_thumb = cv2.resize(render_change_thumb, (thumb_w, thumb_h))
                    cv2.putText(render_change_thumb, "R_change Raw Response", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    panels.append(render_change_thumb)

                if render_change_mask_thumb is not None:
                    if render_change_mask_thumb.shape[0] != thumb_h or render_change_mask_thumb.shape[1] != thumb_w:
                        render_change_mask_thumb = cv2.resize(render_change_mask_thumb, (thumb_w, thumb_h))
                    threshold_label = f"R_change Mask > {float(self.gui_change_threshold.value):.2f}"
                    cv2.putText(render_change_mask_thumb, threshold_label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    panels.append(render_change_mask_thumb)
                    
                if render_upd_thumb is not None:
                    # Enforce exact shape
                    if render_upd_thumb.shape[0] != thumb_h or render_upd_thumb.shape[1] != thumb_w:
                        render_upd_thumb = cv2.resize(render_upd_thumb, (thumb_w, thumb_h))
                    cv2.putText(render_upd_thumb, "Updated Scene", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    panels.append(render_upd_thumb)
                
                if len(panels) > 0:
                    composite = np.concatenate(panels, axis=0) # Vertically stack
                    self.gui_inference_image.image = composite
            
        except Exception as e:
            print(f"Render error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._render_lock.release()
    
    def run(self):
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nShutting down viewer...")


def main():
    parser = argparse.ArgumentParser(description="Interactive Gaussian Splatting Viewer")
    parser.add_argument("--ref_ply", type=str, default=None,
                        help="Path to reference 3DGS point_cloud.ply (inferred from temporal checkpoint)")
    parser.add_argument("--updated_ply", type=str, default=None,
                        help="Path to updated scene PLY (optional)")
    parser.add_argument("--change_ply", type=str, default=None,
                        help="Path to change scene PLY (optional)")
    parser.add_argument("--temporal_checkpoint", type=str, default=None,
                        help="DC-only temporal checkpoint to materialize for inspection")
    parser.add_argument("--temporal_state", type=int, default=0,
                        help="Temporal state slot to inspect (default: 0)")
    parser.add_argument("--temporal_export_ply", type=str, default=None,
                        help="Optional path for the materialized temporal change PLY")
    parser.add_argument("--cameras_json", type=str, default=None,
                        help="Path to cameras.json (auto-detected from ref_ply if not given)")
    parser.add_argument("--inference_dir", type=str, default=None,
                        help="Path to inference images directory (auto-detected if not given)")
    parser.add_argument("--sh_degree", type=int, default=3,
                        help="Spherical harmonics degree (default: 3)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Viser server port (default: 8080)")
    parser.add_argument("--resolution", type=int, default=1,
                        help="Resolution downscale factor (default: 1, matching training)")
    parser.add_argument("--initial_scene", choices=("auto", "reference", "updated", "change"), default="auto",
                        help="Scene selected when the viewer opens")
    parser.add_argument("--change_display", choices=("auto", "raw", "binary"), default="auto",
                        help="Initial R_change visualization (raw is useful for checkpoint inspection)")
    args = parser.parse_args()

    if args.temporal_checkpoint and args.change_ply:
        parser.error("--temporal_checkpoint and --change_ply are mutually exclusive")

    temporal_export = None
    if args.temporal_checkpoint:
        temporal_export = ensure_dc_state_viewer_export(
            args.temporal_checkpoint,
            state_id=args.temporal_state,
            output_ply=args.temporal_export_ply,
        )
        args.change_ply = temporal_export["output_ply"]
        if args.ref_ply is None:
            args.ref_ply = temporal_export["base_ply"]

        config_path = Path(args.temporal_checkpoint).expanduser().resolve().parent / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if args.cameras_json is None and config.get("fixed_cameras_json"):
                args.cameras_json = str(Path(config["fixed_cameras_json"]).expanduser())
            if args.inference_dir is None and config.get("source_path"):
                source_path = Path(config["source_path"]).expanduser()
                if not source_path.is_absolute():
                    source_path = (Path.cwd() / source_path).resolve()
                args.inference_dir = str(source_path / "inference_scene" / "images")

        print("Temporal R_change viewer export:")
        print(f"  state: {temporal_export['state_id']}")
        print(f"  active Gaussians: {temporal_export['active_gaussian_count']:,} / {temporal_export['source_gaussian_count']:,}")
        print(f"  PLY: {temporal_export['output_ply']}")
        print(f"  manifest: {temporal_export['manifest']}")

    if args.ref_ply is None:
        parser.error("--ref_ply is required unless --temporal_checkpoint provides it")

    initial_scene = args.initial_scene
    if initial_scene == "auto":
        initial_scene = "change" if args.temporal_checkpoint else "reference"
    change_display = args.change_display
    if change_display == "auto":
        change_display = "raw" if args.temporal_checkpoint else "binary"
    
    viewer = GaussianViewer(
        ref_ply=args.ref_ply,
        updated_ply=args.updated_ply,
        change_ply=args.change_ply,
        cameras_json=args.cameras_json,
        inference_dir=args.inference_dir,
        sh_degree=args.sh_degree,
        port=args.port,
        resolution=args.resolution,
        initial_scene=initial_scene,
        change_display=change_display,
    )
    viewer.run()


if __name__ == "__main__":
    main()
