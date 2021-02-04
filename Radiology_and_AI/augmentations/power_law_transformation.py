import numpy as np

# This is about 1.2 seconds per iteration, which is probably acceptable
def power_law_transformation(img, seg, gain_distr=[0.8, 1.2], gamma_distr=[0.8, 1.2]):
  # Apply this scaling to all images in series (as opposed to each independently?)
  i = img
  gain = np.random.uniform(low=gain_distr[0], high=gain_distr[1])
  gamma = np.random.uniform(low=gamma_distr[0], high=gamma_distr[1])
  #for channel in i.shape[0]:
  #  i[channel] = _pl(i[channel], gain, gamma)
  i = _pl(i, gain, gamma)
  return i, seg

  # This probably works as intended
def _pl(img, gain, gamma):
  return img**gamma * gain