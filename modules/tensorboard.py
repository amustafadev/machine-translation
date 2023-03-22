# imports
import numpy as np
from torch.utils.tensorboard import SummaryWriter


# class: Tensorboard
class Tensorboard:
  # constructor
  def __init__(self, log_dir: str) -> None:
    # initialize parameters
    self.writer = SummaryWriter(log_dir=log_dir)
    self.steps = {}

  # method: add step
  def add_step(self, tag: str):
    self.steps[tag] = self.steps[tag] + 1 if tag in self.steps else 1
    return self.steps[tag]

  # method: add scalar
  def add_scalar(self, tag: str, value: float):
    self.writer.add_scalar(tag=tag, scalar_value=value, global_step=self.add_step(tag))

  # method: add images
  def add_images(self, tag: str, images: np.ndarray, limit: int = 16, dataformats='NHWC'):
    if len(images) > limit:
      images = images[:limit]
    self.writer.add_images(tag=tag, img_tensor=images, global_step=self.add_step(tag), dataformats=dataformats)

  # method: add text
  def add_text(self, tag: str, text: str):
    self.writer.add_text(tag=tag, text_string=text, global_step=self.add_step(tag))
