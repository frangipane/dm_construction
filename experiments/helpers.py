## Helper Functions
import base64
import tempfile
import textwrap

import matplotlib.pyplot as plt
import dm_construction


def show_rgb_observation(rgb_observation, size=5):
  """Plots a RGB observation, as returned from a Unity environment.

  Args:
  rgb_observation: numpy array of pixels
  size: size to set the figure
  """
  _, ax = plt.subplots(figsize=(size, size))
  ax.imshow(rgb_observation)
  ax.set_axis_off()
  ax.set_aspect("equal")


def print_status(env_, time_step_):
  """Prints reward and episode termination information."""
  status = "r={}, p={}".format(time_step_.reward, time_step_.discount)
  if time_step_.discount == 0:
    status += " (reason: {})".format(env_.termination_reason)
  print(status)


def get_environment(problem_type, wrapper_type="discrete_relative",
                    difficulty=0, curriculum_sample=False):
  """Gets the environment.

  This function separately creates the unity environment and then passes it to
  the environment factory. We do this so that we can add an observer to the
  unity environment to get all frames from which we will create a video.

  Args:
    problem_type: the name of the task
    wrapper_type: the name of the wrapper
    difficulty: the difficulty level
    curriculum_sample: whether to sample difficulty from [0, difficulty]

  Returns:
    env_: the environment
  """
  # Separately construct the Unity env, so we can enable the observer camera
  # and set a higher resolution on it.
  unity_env = dm_construction.get_unity_environment(
    observer_width=600,
    observer_height=600,
    include_observer_camera=True,
    max_simulation_substeps=50)

  # Create the main environment by passing in the already-created Unity env.
  env_ = dm_construction.get_environment(
    problem_type, unity_env, wrapper_type=wrapper_type,
    curriculum_sample=curriculum_sample, difficulty=difficulty)

    # Create an observer to grab the frames from the observer camera.
  env_.core_env.enable_frame_observer()
  return env_


def make_video(frames_):
  """Creates a video from a given set of frames."""
  # Create the Matplotlib animation and save it to a temporary file.
  with tempfile.NamedTemporaryFile(suffix=".mp4") as fh:
    writer = animation.FFMpegWriter(fps=20)
    fig = plt.figure(frameon=False, figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_aspect("equal")
    im = ax.imshow(np.zeros_like(frames_[0]), interpolation="none")
    with writer.saving(fig, fh.name, 50):
      for frame in frames_:
        im.set_data(frame)
        writer.grab_frame()
    plt.close(fig)

    # Read and encode the video to base64.
    mp4 = open(fh.name, "rb").read()
    data_url = "data:video/mp4;base64," + base64.b64encode(mp4).decode()

  # Display the video in the notebook.
  return HTML(textwrap.dedent(
    """
    <video controls>
    <source src="{}" type="video/mp4">
    </video>
    """.format(data_url).strip()))
