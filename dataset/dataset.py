import cv2
import numpy as np
import time
import os

DIRECTORY = "./test/"

if __name__ == "__main__":
    files = os.listdir(DIRECTORY)
    files = [f for f in files if os.path.isfile(os.path.join(DIRECTORY, f))]

    for file in files:
        file_name = str(file)
        filename = DIRECTORY + file_name
        outputname = filename[: len(filename) - 4] + ".txt"
        videoname = filename[: len(filename) - 4] + "_op.mp4"
        print(filename)

        with open(outputname, "w") as file:
            video_capture = cv2.VideoCapture(filename)
            frame_width = int(video_capture.get(3))
            frame_height = int(video_capture.get(4))
            fourcc = cv2.VideoWriter_fourcc(*"m", "p", "4", "v")
            out = cv2.VideoWriter(videoname, fourcc, 30.0, (frame_width, frame_height))
            timestamp = 0
            prev = "0, 0, 0\n"
            time.sleep(3)

            while video_capture.isOpened():
                ret, image = video_capture.read()
                if not ret:
                    break
                timestamp = timestamp + 1
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                lower = np.array([10, 50, 0])
                upper = np.array([50, 150, 255])
                mask_all = cv2.inRange(hsv, lower, upper)
                mask_all = cv2.morphologyEx(
                    mask_all, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
                )
                mask_all = cv2.morphologyEx(
                    mask_all, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8)
                )
                mask2 = cv2.bitwise_not(mask_all)

                min_x, min_y, max_x, max_y = float("inf"), float("inf"), 0, 0
                zero_indices = np.argwhere(mask2 == 0)
                if zero_indices.size > 0:
                    min_y, min_x = zero_indices.min(axis=0)
                    max_y, max_x = zero_indices.max(axis=0)

                if (
                    min_x > 0
                    and min_x < frame_width
                    and max_x < frame_width
                    and min_y > 0
                    and min_y < frame_height
                    and max_y < frame_height
                ):
                    cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
                    xx = (min_x + max_x) // 2
                    yy = (min_y + max_y) // 2
                    area = (max_x - min_x) * (max_y - min_y)
                    area = (
                        area**0.5
                    )  # To reduce error, square root of area has been used.
                    prev = f"{xx}, {yy}, {area}\n"
                    file.write(f"{timestamp}, {xx}, {yy}, {area}\n")
                else:
                    file.write(f"{timestamp}, " + prev)

                out.write(image)

            video_capture.release()
            out.release()

    cv2.destroyAllWindows()
