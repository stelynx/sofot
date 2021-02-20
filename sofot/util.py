import os
from os.path import isdir

import cv2
import numpy
import scipy.io


def get_videos_in_dataset(dataset):
  dataset_path = os.path.join("data", dataset)
  return sorted([name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))])


def create_sofot_dirs_for_dataset_videos(dataset, videos):
  dataset_path = os.path.join("data", f"sofot_{dataset}")
  if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

  for video in videos:
    video_path = os.path.join(dataset_path, video)
    if not os.path.exists(video_path):
      os.mkdir(video_path)
      os.mkdir(os.path.join(video_path, "detections"))


def file_benchmark(dataset):
  return os.path.join("data", f"sofot_{dataset}", "benchmark.csv")


def write_bboxes_for_video_frame(dataset, bboxes, video, frame):
  filename = os.path.join("data", f"sofot_{dataset}", video, "detections", "{}.txt".format((f"0000{frame}")[-5:]))
  with open(filename, mode='w') as f:
    for bbox in bboxes:
      f.write(f"{bbox[0][0]} {bbox[0][1]} {bbox[1][0]} {bbox[1][1]}\n")


def render_filename_for_video(dataset, video):
  return os.path.join("data", f"sofot_{dataset}", video, "output.avi")


def img_name_for_video_frame(dataset, video, frame):
  return os.path.join("data", dataset, video, "images", "{}.jpg".format((f"0000{frame}")[-5:]))


def img_for_video_frame(dataset, video, frame):
  return cv2.imread(img_name_for_video_frame(dataset, video, frame))


def iou(bb1, bb2):
  # Extract points
  p10, p11 = bb1
  p20, p21 = bb2

  # Intersection.
  x0 = float(max(p10[0], p20[0]))
  y0 = float(max(p10[1], p20[1]))
  x1 = float(min(p11[0], p21[0]))
  y1 = float(min(p11[1], p21[1]))
  if x1 < x0 or y1 < y0:
    return 0.0
  intersection_area = (x1 - x0) * (y1 - y0)

  # Areas of original boxes.
  bb1_area = float((p11[0] - p10[0]) * (p11[1] - p10[1]))
  bb2_area = float((p21[0] - p20[0]) * (p21[1] - p20[1]))

  return intersection_area / (bb1_area + bb2_area - intersection_area)


def bbox_distance(bb1, bb2):
  # Only called when overlap is empty!

  # Extract points
  p10, p11 = bb1
  p20, p21 = bb2

  # Calculate minimum distance between borders.
  d = numpy.Inf
  if p11[0] < p20[0]:
    dt = p20[0] - p11[0]
    if dt < d:
      d = dt
  if p10[0] > p21[0]:
    dt = p11[0] - p21[0]
    if dt < d:
      d = dt
  if p11[1] < p20[1]:
    dt = p20[1] - p11[1]
    if dt < d:
      d = dt
  if p10[1] > p21[1]:
    dt = p10[1] - p21[1]
    if dt < d:
      d = dt
  
  return d


def get_data(dataset):
  if dataset == "modd1":
    return get_modd1_data()


def get_modd1_data():
  # Iterate over all 12 videos.
  video_data = []
  for video in range(1, 13):
    path = os.path.join("data", "modd1", "0{}".format(video)[-2:], "gt.mat")
    print("Processing {}...".format(path))

    # Load annotations from MATLAB .mat file.
    contents = scipy.io.loadmat(path)
    largeobjects = contents["largeobjects"]
    smallobjects = contents["smallobjects"]

    # Get number of frames.
    nframes_l = largeobjects.shape[0]
    nframes_s = smallobjects.shape[0]
    assert nframes_l == nframes_s
    nframes = nframes_s

    # Combine large and small objects.
    annotations = [None for _ in range(nframes)]
    for i in range(nframes):
      original_detections = largeobjects[i][0]
      detections = []
      for col in range(original_detections.shape[1]):
        detections.append([int(original_detections[row][col]) for row in range(original_detections.shape[0])])

      original_detections = smallobjects[i][0]
      for col in range(original_detections.shape[1]):
        detections.append([int(original_detections[row][col]) for row in range(original_detections.shape[0])])

      annotations[i] = detections
    
    video_data.append(annotations)

  return video_data