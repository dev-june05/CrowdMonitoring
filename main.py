"""
Crowd Monitoring & Density Estimation System
=============================================
Entry point with CLI argument parsing.

Usage:
    python main.py                            # webcam, default settings (camera 0)
    python main.py --camera 0                 # Explicitly select camera 0 (high-res)
    python main.py --camera 1                 # Select camera 1 (change the number to use other cameras)
    python main.py --source video.mp4         # video file input
    python main.py --roi config/roi.json      # preloaded ROI
    python main.py --grid 12 12               # custom grid resolution
    python main.py --model models/yolov8s.pt  # different YOLO model
    python main.py --roi-mode auto            # auto segmentation mode
    python main.py --no-heatmap               # disable heatmap
"""

import argparse
import sys
from src import PipelineConfig
from src.pipeline import Pipeline


def parse_args():
    """Parse command-line arguments into PipelineConfig."""
    parser = argparse.ArgumentParser(
        description="Intelligent Crowd Monitoring and Density Estimation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Keyboard Controls (during runtime):
  H  Toggle heatmap overlay
  G  Toggle grid lines
  F  Toggle foot points
  V  Toggle velocity arrows
  D  Toggle stats panel
  M  Switch ROI mode (manual/auto)
  R  Redefine ROI polygon
  S  Save screenshot
  Q  Quit
        """
    )

    # Input
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source", type=str, default=None,
        help="Path to a video file (e.g. --source footage.mp4)"
    )
    source_group.add_argument(
        "--camera", type=int, default=None, metavar="INDEX",
        help="Webcam index to use (e.g. --camera 0, --camera 1, --camera 2). Default: 0"
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Capture/display width (default: 1280). Lower = faster. Try 640 for max speed."
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Capture/display height (default: 720). Lower = faster."
    )
    parser.add_argument(
        "--yolo-imgsz", type=int, default=640,
        help="YOLO inference image size (default: 640). Try 320 for faster inference."
    )
    parser.add_argument(
        "--skip-frames", type=int, default=0, metavar="N",
        help="Run detection only every N+1 frames (0=every frame). Use 1 or 2 to boost FPS."
    )
    parser.add_argument(
        "--model", type=str, default="models/yolov8n.pt",
        help="Path to YOLO model weights (default: models/yolov8n.pt)"
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="GPU device index (default: 0)"
    )

    # ROI
    parser.add_argument(
        "--roi", type=str, default="config/roi_config.json",
        help="Path to ROI config JSON file (default: config/roi_config.json)"
    )
    parser.add_argument(
        "--roi-mode", type=str, default="manual", choices=["manual", "auto"],
        help="ROI mode: 'manual' (polygon) or 'auto' (BiSeNetV2 segmentation)"
    )
    parser.add_argument(
        "--redefine-roi", action="store_true",
        help="Force redefine ROI polygon even if config exists"
    )

    # Detection
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold (default: 0.25)"
    )

    # Density
    parser.add_argument(
        "--grid", type=int, nargs=2, default=[8, 8], metavar=("ROWS", "COLS"),
        help="Density grid resolution (default: 8 8)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.3,
        help="KDE adaptive bandwidth: sigma = gamma * bbox_height (default: 0.3)"
    )
    parser.add_argument(
        "--fixed-sigma", type=float, default=20.0,
        help="KDE fixed bandwidth sigma (default: 20.0)"
    )
    parser.add_argument(
        "--no-adaptive", action="store_true",
        help="Use fixed-sigma KDE instead of adaptive"
    )

    # Temporal
    parser.add_argument(
        "--ema-alpha", type=float, default=0.3,
        help="EMA smoothing alpha (default: 0.3)"
    )

    # Congestion
    parser.add_argument(
        "--warn-threshold", type=float, default=3.0,
        help="Congestion WARNING threshold (people/cell, default: 3.0)"
    )
    parser.add_argument(
        "--critical-threshold", type=float, default=6.0,
        help="Congestion CRITICAL threshold (people/cell, default: 6.0)"
    )

    # Visualization
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="Disable heatmap overlay"
    )
    parser.add_argument(
        "--heatmap-opacity", type=float, default=0.4,
        help="Heatmap overlay opacity (default: 0.4)"
    )
    parser.add_argument(
        "--show-grid", action="store_true",
        help="Show grid lines by default"
    )
    parser.add_argument(
        "--show-velocity", action="store_true",
        help="Show velocity arrows by default"
    )

    # Auto segmentation
    parser.add_argument(
        "--seg-config", type=str, default="",
        help="Path to BiSeNetV2 config file (for auto ROI mode)"
    )
    parser.add_argument(
        "--seg-checkpoint", type=str, default="",
        help="Path to BiSeNetV2 checkpoint file (for auto ROI mode)"
    )
    parser.add_argument(
        "--seg-interval", type=int, default=30,
        help="Run segmentation every N frames (default: 30)"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directory for screenshots and recordings (default: outputs)"
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Record output video"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Resolve source: explicit camera index, video file, or default camera 0
    if args.source is not None:
        resolved_source = args.source          # video file path
    elif args.camera is not None:
        resolved_source = args.camera          # webcam integer index
    else:
        resolved_source = 0                    # default: camera index 0

    # Build config from parsed arguments
    config = PipelineConfig(
        source=resolved_source,
        model_path=args.model,
        device=args.device,
        capture_width=args.width,
        capture_height=args.height,
        yolo_imgsz=args.yolo_imgsz,
        skip_frames=args.skip_frames,
        roi_config_path=args.roi,
        roi_mode=args.roi_mode,
        redefine_roi=args.redefine_roi,
        confidence_threshold=args.conf,
        person_class_id=0,
        grid_rows=args.grid[0],
        grid_cols=args.grid[1],
        kde_gamma=args.gamma,
        kde_fixed_sigma=args.fixed_sigma,
        use_adaptive_kde=not args.no_adaptive,
        ema_alpha=args.ema_alpha,
        use_adaptive_alpha=True,
        warning_threshold=args.warn_threshold,
        critical_threshold=args.critical_threshold,
        heatmap_opacity=args.heatmap_opacity,
        show_heatmap=not args.no_heatmap,
        show_grid=args.show_grid,
        show_foot_points=True,
        show_roi=True,
        show_velocity=args.show_velocity,
        show_stats=True,
        seg_model_config=args.seg_config,
        seg_model_checkpoint=args.seg_checkpoint,
        seg_run_interval=args.seg_interval,
        output_dir=args.output_dir,
        record=args.record,
    )

    print("=" * 60)
    print("  Crowd Monitoring & Density Estimation System")
    print("=" * 60)
    if isinstance(config.source, int):
        print(f"  Source:     Camera index {config.source}")
    else:
        print(f"  Source:     {config.source}")
    print(f"  Model:      {config.model_path}")
    print(f"  ROI Mode:   {config.roi_mode}")
    print(f"  Grid:       {config.grid_rows}x{config.grid_cols}")
    print(f"  KDE:        {'adaptive (γ=' + str(config.kde_gamma) + ')' if config.use_adaptive_kde else 'fixed (σ=' + str(config.kde_fixed_sigma) + ')'}")
    print(f"  EMA Alpha:  {config.ema_alpha}")
    print(f"  Thresholds: WARN={config.warning_threshold} CRIT={config.critical_threshold}")
    print("=" * 60)
    print()

    # Create and run pipeline
    pipeline = Pipeline(config)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
