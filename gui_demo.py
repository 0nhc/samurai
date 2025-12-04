import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

# Color list for multi-object visualization
color = [
    (255, 0, 0),   # Blue (obj_id 0)
    (0, 255, 0),   # Green (obj_id 1)
    (0, 0, 255),   # Red (obj_id 2)
    (255, 255, 0), # Cyan
    (255, 0, 255), # Magenta
    (0, 255, 255), # Yellow
    (255, 128, 0), # Orange
    (128, 0, 255), # Purple
]

# Global variables for GUI
drawing = False
start_point = None
end_point = None
bboxes = []
current_bbox = None
window_name = "Select Bounding Boxes - Press 'q' when done, 'r' to reset, 'd' to delete last"


def mouse_callback(event, x, y, flags, param):
    """Mouse callback for drawing bounding boxes."""
    global drawing, start_point, end_point, current_bbox, bboxes
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)
        current_bbox = None
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        if start_point and end_point:
            # Ensure x1 < x2 and y1 < y2
            x1 = min(start_point[0], end_point[0])
            y1 = min(start_point[1], end_point[1])
            x2 = max(start_point[0], end_point[0])
            y2 = max(start_point[1], end_point[1])
            
            # Only add if box has some area
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                bbox = (x1, y1, x2, y2)
                bboxes.append(bbox)
                print(f"Added bounding box {len(bboxes)}: {bbox}")
            start_point = None
            end_point = None
            current_bbox = None


def draw_bboxes_on_frame(frame, bboxes_list, current_start=None, current_end=None):
    """Draw all bounding boxes on the frame."""
    frame_copy = frame.copy()
    
    # Draw completed bounding boxes
    for idx, bbox in enumerate(bboxes_list):
        x1, y1, x2, y2 = bbox
        color_idx = idx % len(color)
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color[color_idx], 2)
        cv2.putText(frame_copy, f"Obj {idx}", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color[color_idx], 2)
    
    # Draw current bounding box being drawn
    if current_start and current_end:
        x1 = min(current_start[0], current_end[0])
        y1 = min(current_start[1], current_end[1])
        x2 = max(current_start[0], current_end[0])
        y2 = max(current_start[1], current_end[1])
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    # Draw instructions
    cv2.putText(frame_copy, "Click and drag to select objects", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_copy, "Press 'q' when done, 'r' to reset, 'd' to delete last", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_copy, f"Selected: {len(bboxes)} objects", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame_copy


def select_bboxes_gui(first_frame):
    """GUI for selecting bounding boxes on the first frame."""
    global drawing, start_point, end_point, bboxes, current_bbox
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n=== Bounding Box Selection ===")
    print("Instructions:")
    print("  - Click and drag to draw a bounding box")
    print("  - Press 'q' to finish selection and start tracking")
    print("  - Press 'r' to reset all boxes")
    print("  - Press 'd' to delete the last box")
    print("  - Press 'ESC' to exit\n")
    
    while True:
        display_frame = draw_bboxes_on_frame(first_frame, bboxes, start_point, end_point)
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            if len(bboxes) > 0:
                print(f"\nStarting tracking with {len(bboxes)} objects...")
                break
            else:
                print("Please select at least one bounding box!")
        
        elif key == ord('r'):
            bboxes = []
            print("Reset all bounding boxes")
        
        elif key == ord('d'):
            if len(bboxes) > 0:
                removed = bboxes.pop()
                print(f"Removed bounding box: {removed}")
            else:
                print("No bounding boxes to remove")
        
        elif key == 27:  # ESC
            print("Exiting...")
            cv2.destroyAllWindows()
            sys.exit(0)
    
    cv2.destroyAllWindows()
    return bboxes


def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")


def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")


def main(args):
    # Load first frame for GUI selection
    print(f"Loading video: {args.video_path}")
    if osp.isdir(args.video_path):
        frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) 
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"))])
        if not frames:
            raise ValueError("No image frames found in the directory.")
        first_frame = cv2.imread(frames[0])
        if first_frame is None:
            raise ValueError(f"Could not load first frame from {frames[0]}")
    else:
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {args.video_path}")
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Could not read first frame from video.")
        cap.release()
    
    height, width = first_frame.shape[:2]
    print(f"Video dimensions: {width}x{height}")
    
    # GUI for selecting bounding boxes
    selected_bboxes = select_bboxes_gui(first_frame)
    
    if len(selected_bboxes) == 0:
        print("No bounding boxes selected. Exiting.")
        return
    
    # Convert bboxes from (x1, y1, x2, y2) to format expected by predictor: (x1, y1, x2, y2)
    initial_bboxes = selected_bboxes.copy()
    print(f"\nSelected {len(initial_bboxes)} bounding boxes:")
    for idx, bbox in enumerate(initial_bboxes):
        print(f"  Object {idx}: {bbox}")
    
    # Initialize predictor
    model_cfg = determine_model_cfg(args.model_path)
    print(f"\nLoading model from {args.model_path}...")
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
    frames_or_path = prepare_frames_or_path(args.video_path)
    
    # Load all frames for efficient processing
    print("\nLoading video frames...")
    frame_rate = 30
    loaded_frames = []
    if osp.isdir(args.video_path):
        frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) 
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"))])
        loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
        total_frames = len(loaded_frames)
    else:
        cap = cv2.VideoCapture(args.video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        if frame_rate <= 0:
            frame_rate = 30
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            loaded_frames.append(frame)
        cap.release()
        total_frames = len(loaded_frames)
    
    if len(loaded_frames) == 0:
        raise ValueError("No frames loaded from video.")
    
    print(f"Loaded {total_frames} frames at {frame_rate:.2f} FPS")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, frame_rate, (width, height), isColor=True)
    
    # Pure green background
    green_background = np.zeros((height, width, 3), dtype=np.uint8)
    green_background[:, :] = (0, 255, 0)  # Pure green (BGR format)
    
    print("\nStarting tracking...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        
        # Initialize all objects
        print("Initializing predictor with selected objects...")
        for obj_id, bbox in enumerate(initial_bboxes):
            # bbox is (x1, y1, x2, y2), predictor expects same format
            print(f"  Adding object {obj_id} with bbox {bbox}")
            predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=obj_id)
        
        print("Propagating masks through video...")
        frame_count = 0
        combined_mask = np.zeros((height, width), dtype=bool)
        
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            if frame_idx >= len(loaded_frames):
                print(f"Warning: frame_idx {frame_idx} exceeds loaded frames. Skipping.")
                continue
            
            # Create output frame with green background
            output_frame = green_background.copy()
            original_frame = loaded_frames[frame_idx]
            
            # Combine all masks
            combined_mask.fill(False)
            for obj_id, mask in zip(object_ids, masks):
                # Handle mask shape
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                
                # Ensure mask matches video dimensions
                mask = mask > 0.0
                if mask.shape != (height, width):
                    mask_uint8 = (mask.astype(np.uint8) * 255)
                    mask = cv2.resize(mask_uint8, (width, height), interpolation=cv2.INTER_NEAREST) > 127
                
                # Combine masks using OR
                combined_mask = np.logical_or(combined_mask, mask)
            
            # Apply combined mask: copy masked pixels from original frame to output
            mask_3d = np.stack([combined_mask] * 3, axis=2)  # Convert to 3-channel mask
            output_frame = np.where(mask_3d, original_frame, output_frame)
            
            out.write(output_frame)
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames...", end='\r')
        
        print(f"\n  Processed {frame_count} frames")
    
    out.release()
    
    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
    
    print(f"\nTracking complete! Output video saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUI-based object tracking with green background")
    parser.add_argument("--video_path", required=True, 
                       help="Input video path or directory of frames.")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", 
                       help="Path to the model checkpoint.")
    parser.add_argument("--output_path", default="output_green_background.mp4", 
                       help="Path to save the output video with green background.")
    args = parser.parse_args()
    main(args)

