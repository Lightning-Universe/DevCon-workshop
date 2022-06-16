# Bonus: Multi-node DDP with Lightning

Here we want to quickly tease the experimental features for multi-node distributed LightningWork!
The code demonstrates how to launch several instances of LightningWork and connecting them through the PyTorch 
distributed backend to all train together as a cluster of nodes, seamlessly with ZERO hardware configuration.


## Running the Example

You can simulate DDP locally to make sure everything works before moving to the cloud:

```commandline
cd 3-Distributed
lightning run app app/app.py
```

The logs should show the following output:

```
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/2
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/2
All distributed processes registered. Starting with 2 processes
```

If everything looks as shown here, you are ready to launch in the cloud!

```commandline
lightning run app app/app.py --cloud
```
