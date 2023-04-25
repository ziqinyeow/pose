"""Beta Version"""

import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .func.image import draw_prediction_on_image
from .func.video import (
    to_gif,
    init_crop_region,
    determine_crop_region,
    run_inference_with_crop_region,
)


def viz(media, model, show=True, save=False, save_path=None):
    if len(media.shape) == 3:
        save_path = save_path or "image.png"
        viz_image(media, model, show, save, save_path)
    else:
        save_path = save_path or "video.gif"
        viz_video(media, model, show, save, save_path)


def viz_image(image, model, show=True, save=False, save_path="image.png"):
    """
    image:                  np.ndarray  -> shape: [width, height, channel]
    model:                  tf/pt model
    show:                   bool        -> show popup media
    save:                   bool        -> save the media file
    save_path:              str         -> if save == True, media will get saved to save_path
    """
    keypoints_with_scores = model(image)
    display_image = tf.expand_dims(image, axis=0)
    display_image = tf.cast(
        tf.image.resize_with_pad(display_image, 1280, 1280),
        dtype=tf.int32,
    )
    output_overlay = draw_prediction_on_image(
        np.squeeze(display_image.numpy(), axis=0),
        keypoints_with_scores,
    )
    plt.figure(figsize=(5, 5))
    plt.imshow(output_overlay)
    _ = plt.axis("off")
    if show:
        plt.show()

    if save:
        plt.savefig(save_path)


def viz_video(video, model, show=False, save=True, save_path="video.gif"):
    """
    video:                  np.ndarray  -> shape: [frameNo, width, height, channel]
    model:                  tf/pt model
    show:                   bool        -> show popup media
    save:                   bool        -> save the media file
    save_path:              str         -> if save == True, media will get saved to save_path
    """
    input_size = model.get_config("input_size")
    num_frames, image_height, image_width, _ = video.shape
    crop_region = init_crop_region(image_height, image_width)

    output_images = []
    for frame_idx in tqdm.tqdm(range(num_frames)):
        keypoints_with_scores = run_inference_with_crop_region(
            model,
            video[frame_idx, :, :, :],
            crop_region,
            crop_size=[input_size, input_size],
        )
        output_images.append(
            draw_prediction_on_image(
                video[frame_idx, :, :, :].numpy().astype(np.int32),
                keypoints_with_scores,
                crop_region=None,
                close_figure=True,
                output_image_height=300,
            )
        )
        crop_region = determine_crop_region(
            keypoints_with_scores, image_height, image_width
        )

    # Prepare gif visualization.
    output = np.stack(output_images, axis=0)
    to_gif(output, path=save_path, fps=25)
