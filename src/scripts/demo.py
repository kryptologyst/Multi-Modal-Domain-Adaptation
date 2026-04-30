"""Demo script launcher."""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Launch demo applications")
    
    parser.add_argument(
        "demo_type",
        choices=["streamlit", "gradio"],
        help="Type of demo to launch"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the demo on"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to run the demo on"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main demo launcher function."""
    args = parse_args()
    
    if args.demo_type == "streamlit":
        demo_path = Path("demo/streamlit_demo.py")
        if not demo_path.exists():
            print(f"Error: Demo file not found at {demo_path}")
            sys.exit(1)
        
        cmd = [
            "streamlit", "run", str(demo_path),
            "--server.port", str(args.port),
            "--server.address", args.host
        ]
        
        print(f"Launching Streamlit demo on {args.host}:{args.port}")
        subprocess.run(cmd)
    
    elif args.demo_type == "gradio":
        print("Gradio demo not implemented yet")
        sys.exit(1)


if __name__ == "__main__":
    main()
