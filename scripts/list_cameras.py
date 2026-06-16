"""
Camera Discovery Utility
========================
Lists all available cameras connected to the system with their indices.
Run this to find out which camera index to use with --camera.

Usage:
    python scripts/list_cameras.py
"""

import cv2

def list_cameras(max_index=10):
    print("Scanning for connected cameras...\n")
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  [camera {idx}]  {w}x{h} @ {fps:.0f} FPS")
            found.append(idx)
            cap.release()

    if not found:
        print("  No cameras found.")
    else:
        print(f"\n{len(found)} camera(s) found.")
        print(f"\nTo use camera {found[0]}, run:")
        print(f"    python main.py --camera {found[0]}")
        if len(found) > 1:
            print(f"\nTo use camera {found[1]}, run:")
            print(f"    python main.py --camera {found[1]}")

if __name__ == "__main__":
    list_cameras()
