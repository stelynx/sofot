import cv2
import numpy

import util


LK_PARAMS = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ST_PARAMS = dict( maxCorners   = 20,
                  qualityLevel = 0.15,
                  minDistance  = 1,
                  blockSize    = 1 )


class Sofot():
  def __init__(self, video, annotations, lk_params=LK_PARAMS, st_params=ST_PARAMS, debug=False):
    self.annotations = annotations
    self.video = video
    self.lk_params = lk_params
    self.st_params = st_params
    self._debug = debug

    self.current_frame = 1
    self.nframes = len(annotations)
    self.color = numpy.random.randint(0, 255, (100, 3))

    self.bboxes = [[(annotations[0][bbi][0], annotations[0][bbi][1]), (annotations[0][bbi][2], annotations[0][bbi][3])] for bbi in range(len(annotations[0]))]
    self._bbox_color = numpy.random.randint(0, 255, (len(self.bboxes), 3))

  def track(self):
    print(f"[Sofot.track] started tracking video {self.video}")
    # Read starting frame and convert it to grayscale.
    frame = util.img_for_video_frame(self.video, self.current_frame)
    self.__frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate ROI masks and good optical flow points.
    of_points = []
    for roi in self.annotations[0]:
      mask = numpy.zeros_like(self.__frame_gray)
      mask[roi[1]:roi[3],roi[0]:roi[2]] = 255
      of_points.append(cv2.goodFeaturesToTrack(self.__frame_gray, mask=mask, **self.st_params))
    self.of_points = of_points

    # Instatiate a mask for displaying optical flow on the image.
    self.__disp_flow_mask = numpy.zeros_like(frame)

    # Error is increased when a detection does not match a ground truth box.
    self.error = 0
    self.frame_with_first_error = None

    # Main loop.
    while self.current_frame < self.nframes:
      self._step()

    # Close all windows at the end.
    cv2.destroyAllWindows()

    # Print errors to the console.
    print(f"[Sofot.track] finished with error {self.error}" + (f" and first frame with error {self.frame_with_first_error}" if self.error > 0 else ""))

  def _step(self):
    # Go to next frame.
    self.current_frame += 1

    # Read frame and convert it to grayscale.
    frame = util.img_for_video_frame(self.video, self.current_frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow for each object.
    of_points_new = []
    bboxes_new = []
    for of_points in self.of_points:
      if len(of_points) == 0:
        continue

      points, st, _ = cv2.calcOpticalFlowPyrLK(self.__frame_gray, frame_gray, of_points, None, **self.lk_params)

      # Select good points.
      of_points_good_new = points[st==1]
      of_points_good_old = of_points[st==1]
      if (len(of_points_good_new) == 0):
        continue

      # Draw optical flow tracks.
      if self._debug:
        for i,(new,old) in enumerate(zip(of_points_good_new, of_points_good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            self.__disp_flow_mask = cv2.line(self.__disp_flow_mask, (a,b), (c,d), self.color[i].tolist(), 2)
            frame = cv2.circle(frame, (a,b), 5, self.color[i].tolist(), -1)

      # Calculate and draw bounding box as [(x0, y0), (x1, y1)].
      xs, ys = zip(*of_points_good_new)
      bbox = [(min(xs), min(ys)), (max(xs), max(ys))]
      frame = cv2.rectangle(frame, bbox[0], bbox[1], color=self._bbox_color[len(bboxes_new)].tolist(), thickness=2)

      # Add frame to image.
      img = cv2.add(frame, self.__disp_flow_mask)

      # Add newly calculated points.
      of_points_new.append(of_points_good_new.reshape(-1, 1, 2))
      bboxes_new.append(bbox)

    # Create pairs of bounding boxes.
    pairs = self._pair_bboxes(self.bboxes, bboxes_new, change_colors=True)

    if self._debug:
      print(self._bbox_color)
      print(pairs)

    # Compare bounding boxes against ground truth. It is ok, if a bounding box is in ground truth
    # and not in detections, however every detection must have a corresponding bounding box.
    i_annot = self.current_frame - 1
    bboxes_ground_truth = [[(bb[0], bb[1]), (bb[2], bb[3])] for bb in self.annotations[i_annot]]
    pairs_ground_truth = self._pair_bboxes(bboxes_new, bboxes_ground_truth)
    diff = len(pairs) - len(pairs_ground_truth)
    if diff > 0:  # cannot be negative
      self.error += diff
      if self.frame_with_first_error is None:
        self.frame_with_first_error = self.current_frame

    if self._debug:
      print(pairs_ground_truth)
      print("error", self.error)

    # Write error on screen.
    img = cv2.putText(img, f"Error: {self.error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)

    # Display image.
    cv2.imshow("SOFOT", img)
    cv2.waitKey(0 if self._debug else 1)

    # Update previous gray frame, previous optical flow points, and previous bboxes.
    self.__frame_gray = frame_gray.copy()
    self.of_points = of_points_new
    self.bboxes = bboxes_new

  def _pair_bboxes(self, bboxes_true, bboxes_new, change_colors=False):
    pairs = []
    for i_true, bb_true in enumerate(bboxes_true):
      match = None

      # Calculate overlap of bounding boxes.
      iou_max = 0.0
      for i_new, bb_new in enumerate(bboxes_new):
        iou = util.iou(bb_true, bb_new)
        if iou > iou_max:
          iou_max = iou
          match = i_new

      if match is not None:
        pairs.append((i_true, match))
      
    # Check for missing.
    if len(pairs) < len(bboxes_true):
      matched_true = [x[0] for x in pairs]
      matched_new = [x[1] for x in pairs]
      for i_true, bb_true in enumerate(bboxes_true):
        if i_true in matched_true:
          continue

        match = None
        d_min = numpy.Inf
        for i_new, bb_new in enumerate(bboxes_new):
          # Skip already matched new bounding boxes.
          if i_new in matched_new:
            continue

          d = util.bbox_distance(bb_true, bb_new)
          if d < d_min:
            d_min = d
            match = i_new

        if match is not None:
          pairs.append((i_true, match))
          matched_true.append(i_true)

      # Update colors if still not all matched.
      if change_colors and len(pairs) < len(bboxes_true):
        colors_new = []
        for ci in [x[0] for x in sorted(pairs, key=lambda x: x[1])]:
          colors_new.append(self._bbox_color[ci])
        self._bbox_color = numpy.array(colors_new)

    return pairs
