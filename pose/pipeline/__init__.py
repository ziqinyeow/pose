from ..model import AutoModel
from ..utils.plot import plot
from ..utils.transform import resize
from ..utils.crop import init_crop_region, determine_crop_region

import cv2
import typing
from vidgear.gears import CamGear, WriteGear

def pipeline(
    model: str,
    src: str,
    show: bool = True,
    save: typing.Union[bool, str] = False,
):
    m = AutoModel.from_pretrained(model)
    
    stream = CamGear(source=src, stream_mode=src.startswith("https://"), colorspace="COLOR_BGR2RGB").start()

    # framerate = stream.framerate
    framecount = int(stream.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    # image
    if framecount <= 1:
        frame = stream.read()
        keypoints = m(frame)

        # resize to square -> prevent scaling offset -> keypoints misalignment
        frame = resize(frame)
        plot(frame, keypoints, conf_thres=0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if show:
            cv2.imshow("output", frame)
            cv2.waitKey(0)
        
        if isinstance(save, (bool, str)):
            if isinstance(save, bool) and save:
                filename = src.split("/")[-1]
                filename = filename.split(".")[0] + "_inference." + filename.split(".")[1]
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

    while True:
        frame = stream.read()
        if frame is None:
            break

        keypoints = m(frame, crop_region)
        plot(frame, keypoints, 3, side="both", conf_thres=0.2)
        crop_region = determine_crop_region(keypoints, H, W)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if show:
            cv2.imshow("output", frame)

        if save:
            writer.write(frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()
        
    