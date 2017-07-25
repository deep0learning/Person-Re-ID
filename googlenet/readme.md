# Dataset Dir

In the configuration file `Config.py`, change the following path to your directory of Market1501.
```python
dataset_root = os.path.expanduser('~/Dataset/Market-1501-v15.09.15')
```

# Training

Prerequisite: You have to install python package cv2 (for resizing images) by `pip install opencv-python`.

In the shell, go to project directory, and run command `python -u train_reid_res50.py`. According to the current setting in `Config.py`, it will train the res50 on all training data of Market1501 for 80 epochs, saving checkpoint for every 5 epochs.

# Testing

Prerequisite: You have to install a reid python package [open-reid](https://github.com/Cysu/open-reid) for evaluation.

In the shell, go to project directory, and run command `python -u test_reid_res50.py`. According to the current setting in `Config.py`, it will test on the saved checkpoints.
