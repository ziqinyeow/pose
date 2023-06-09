from ..model import AutoModel
from ..utils.plot import plot
from ..utils.transform import resize
from ..utils.crop import init_crop_region, determine_crop_region

import cv2
import time
import typing
from vidgear.gears import CamGear, WriteGear


def pipeline(
    model: str,
    src: typing.Union[str, int],
    show: bool = True,
    save: typing.Union[bool, str] = False,
    side: str = "both",
    angle: bool = True,
):
    m = AutoModel.from_pretrained(model)

    stream = CamGear(
        source=src,
        stream_mode=str(src).startswith("https://"),
        colorspace="COLOR_BGR2RGB",
    ).start()

    # framerate = stream.framerate
    framecount = int(stream.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    # image
    if framecount <= 1 and not isinstance(src, int):
        frame = stream.read()
        keypoints = m(frame)

        # resize to square -> prevent scaling offset -> keypoints misalignment
        if m.config.get("resize") is not None and m.config.get("resize"):
            frame = resize(frame)

        plot(frame, keypoints, steps=m.config.get("steps", 3), conf_thres=0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if show:
            cv2.imshow("output", frame)
            cv2.waitKey(0)

        if isinstance(save, (bool, str)):
            if isinstance(save, bool) and save:
                filename = src.split("/")[-1]
                filename = (
                    filename.split(".")[0] + "_inference." + filename.split(".")[1]
                )
                cv2.imwrite(filename, frame)
            elif save:
                cv2.imwrite(save, frame)

        cv2.destroyAllWindows()
        stream.stop()
        return

    if isinstance(save, (bool, str)):
        if isinstance(save, bool) and save:
            filename = src.split("/")[-1]
            filename = filename.split(".")[0] + "_inference." + filename.split(".")[1]
            writer = WriteGear(output=filename)
        elif save:
            writer = WriteGear(output=save)

    H = int(stream.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(stream.stream.get(cv2.CAP_PROP_FRAME_WIDTH))

    crop_region = init_crop_region(H, W)

    idx = 0
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        frame = stream.read()
        if frame is None:
            break

        try:
            frame.flags.writeable = False
            if m.config.get("crop_region") is not None and m.config.get("crop_region"):
                keypoints = m(frame, crop_region)
                crop_region = determine_crop_region(keypoints, H, W)
            else:
                keypoints = m(frame)
            frame.flags.writeable = True

            plot(
                frame,
                keypoints,
                steps=m.config.get("steps", 3),
                side=side,
                show_angle=angle,
                conf_thres=0.2,
            )

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            cv2.putText(
                frame,
                f"{idx}/{framecount}",
                (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3,
            )
            cv2.putText(
                frame,
                f"{round(fps, 2)} FPS",
                (100, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3,
            )

            idx += 1
            if show:
                cv2.imshow("output", frame)

            if save:
                writer.write(frame)
        except:
            pass

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()
