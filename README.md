# pull-up tracker

- a custom trained object detection YOLO (v. 5) model that can recognize people doing pull-ups with excellent results:
  ![PR_curve](https://github.com/davide710/pull-up_tracker/assets/106482229/158e7e03-ba4a-4068-a88b-59bb9e8de809)
- a script (counting_single.py) that, given a video of one person training, can live count how many repetitions are being done
- a script (challenge.py) that tracks (with custom developed class Tracker()) how many pull-ups are executed by each person in the video, as in a competition

### Usage
```
git clone https://github.com/davide710/pull-up_tracker.git
cd pull-up_tracker
pip install numpy opencv-contrib-python pyyaml

# To execute a script with a video
python3 [script.py] /path/to/video
# To execute the script challenge.py with the example
python3 challenge.py videos/pullups_challenge.mp4
# To execute it with live webcam (you may need to adjust webcam configurations in the script)
python3 [script.py] -live
```
This is the result obtained with [videos/pullups_challenge.mp4](videos/pullups_challenge.mp4):
![result](demo_result.avi)

The original video was taken from https://www.youtube.com/shorts/ft7VmEyvcuc?feature=share
#
#
_"Manually counting repetitions? Yeah, I think the ancient Greeks used to do it too"_
