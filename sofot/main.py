import argparse

import sofot
import util


def main(args):
  # Read dataset annotations.
  data = util.get_data(args.dataset)

  # Get video names.
  videos = []
  if args.video is not None:
    videos = [args.video]
  else:
    videos = util.get_videos_in_dataset(args.dataset)

  # Create directory structure for output.
  util.create_sofot_dirs_for_dataset_videos(args.dataset, videos)

  # Run SOFOT on all videos in dataset.
  for video_idx, video in enumerate(videos):
    ot = sofot.Sofot(dataset=args.dataset, video=video, annotations=data[video_idx], benchmark=args.benchmark, render=args.render, save_bbox=args.save_bbox, debug=args.debug)
    ot.track()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="SOFOT - Simple Optical Flow-based Object Tracker")
  parser.add_argument("--dataset", required=True, type=str, help="Folder name in \"data\" folder to be used.")
  parser.add_argument("--video", required=False, type=str, help="Run only on this video.")
  parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode, rendering, saving, and debugging is disabled.")
  parser.add_argument("--render", action="store_true", help="Render video of frames with bounding boxes.")
  parser.add_argument("--save-bbox", action="store_true", help="Save bounding boxes in files for each frame. Format of bounding box is \"x1 y1 x2 y2\" (upper left and lower right corner).")
  parser.add_argument("--debug", action="store_true", help="Run in debug mode. Stops after each frame is presented and prints debugging information to stdout.")

  args = parser.parse_args()

  main(args)
  
