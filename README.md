# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goal of this project is to build a pipeline to detect vehicles in a video. The whole process is divided into two steps:
1. Given a set of car and not-car images, train a classifier to recognize whether an image is a car.
2. For each frame of the video, slide the searching windows with certain scales to detect if the car is contained in these windows. Finally, combine the positive windows and output the bounding box around each car in the frame.

For the detailed implementation, see `report.md`

### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [OpenCV](http://opencv.org/)
- [Scikit-learn](http://scikit-learn.org/stable/)


## Results
The test image results are in folder `test_images`, the video results are in folder `test_videos`


Here's a [link to my project video result](./test_videos/project_video_result.mp4)


<a href="http://www.youtube.com/watch?feature=player_embedded&v=30zLSV16T28
" target="_blank"><img src="./report_images/video_image.jpg"
alt="Video Track 1" width="640" height="360" border="1" /></a>
