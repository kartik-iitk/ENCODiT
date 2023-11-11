# RoboCup MSL Ball Dataset

- The dataset has two folders, `train` and `test` corresponding to straight and non-straight trajectories of the ball.
- `Test` folder contains 3 folders each named corresponding to the distance of the camera from the ball.
- Inside each folder are videos labelled as: `Str_<force by which ball was kicked>_<camera angle around Y axis>.mp4`
- Camera angle around X and Z axis is kept fixed for all videos.
- Recorded time series data in the order `timestamp, x_coordinate, y_coordinate, sqrt(area)` for video `<video_name>.mp4` is in the file `<video_name>.txt` in the same directory.
