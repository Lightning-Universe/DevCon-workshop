# Supercharge Your Training With the PyTorchLightning App

## Preparation

As before, we will copy the existing PyTorch Lightning project to this section folder to keep things cleanly separated:

```commandline
cp -r project 2-PyTorch-Lightning-App/
cd 2-PyTorch-Lightning-App
```

## The PL App

To generate a brand-new app wrapping your PyTorch Lightning script, run:

```commandline
lightning init pl-app project/train_cifar.py
```

Check the current working directory for a folder called `pl-app`. Don't be shy, look inside!

Run the app:

```commandline
lightning run app pl-app/app.py
```
