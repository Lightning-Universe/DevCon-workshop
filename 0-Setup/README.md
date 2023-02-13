# Setup

## Installation

Follow the [steps in the documentation](https://lightning.ai/lightning-docs/installation.html) to get Lightning installed on your computer.
**Important:** Install using a [virtual environment](https://lightning.ai/lightning-docs/install_beginner.html)! Python 3.8 and higher are supported.

Verify the installation by running the test app in this folder.

```commandline
cd 0-Setup
lightning run app test/app.py
```

The output in the terminal should show this:

```
INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view

Hello
```

For later, we will also need the following packages. Let's install them now:

```commandline
pip install pytorch-lightning torchvision torchmetrics streamlit
```

## Getting ready for the cloud

With the installation working locally, let's hop on the cloud. Simply run

```commandline
lightning run app test/app.py --cloud
```

and a browser window will open. On the first time, you will be asked to create a Lightning AI account. Follow the steps on the screen.
