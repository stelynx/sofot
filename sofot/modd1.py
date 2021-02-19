import sofot
import util


if __name__ == "__main__":
  data = util.get_modd1_data()

  video_idx = 0

  ot = sofot.Sofot(video=video_idx+1, annotations=data[video_idx], debug=True)
  ot.track()
