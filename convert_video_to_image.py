import cv2

EXTRACT_FREQUENCY = 1

video_subs = (2, 3, 7, 8, 10, 12, 13, 14, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 31)

for sub in video_subs:
    video_path = f'/home/nieyang/MWRawVideos/MW-18Mar-{sub}.avi'

    video = cv2.VideoCapture()
    if not video.open(video_path):
        raise ValueError("can not open the video")

    count = 1
    while True:
        ret, frame = video.read()
        if not ret:
            break

        if count % EXTRACT_FREQUENCY == 0:
            save_path = f'/home/nieyang/MW_R/Mar{sub}_{count:>06d}.jpg'
            cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

        count += 1

    video.release()
