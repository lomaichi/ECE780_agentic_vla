"""Code shared by the pixel-Q correctors (V1/ and V2/).

  scene   LIBERO plumbing: start states and the training / held-out split guard, target lookup,
          segmentation, camera geometry, gripper pose helpers, the pi0 websocket client and the
          LiberoScene base class (env construction, target, object tilt)
  fcn     the fully convolutional Q-network trunk (stride 8: one output cell per 8x8 pixel window)
  runlog  JSONL logs, Weights & Biases helpers and the actor process pool

V1/ and V2/ entry scripts put this folder's parent (pixel_Q/) on sys.path and import `shared.<module>`.
Python 3.8 compatible (the LIBERO client venv); LIBERO, robosuite, torch and wandb are imported lazily.
"""
