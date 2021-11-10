## Global scale landscape evolution simulations

The *simulation* folder provides some examples to run `goSPL` model at global scale with different approaches to constrain the simulation with paleo-elevation model. 

All these models are approximately 10km resolution and are ran on HPC. A `pbs` cript is also provided (mainly for USYD users running it on Artemis HPC). For the considered resolution a number of CPUs between 80 and 128 seems to be the most appropriate for performance. 

## Constrained model

The first example uses directly the outputs obtained from the *inputgen* folder and forces the model to match with paleo-elevation. 

An example of input file for this type of approach is provided:

- `constrained.yml`

Such simulation will look like the one below where every 5 Ma the model is forced to fit with a paleo-elevation model.

https://user-images.githubusercontent.com/7201912/141037075-368e0694-3e94-486d-ac89-0778646cbb80.mov

## Unconstrained models

Several unconstrained model approaches are also explained.

### First method

In its simplest form, the simulation could be forced using the *geometrical* solution only over time. This will produce results similar to the one below: 

https://user-images.githubusercontent.com/7201912/141037105-728b92df-e8e8-41b9-ad75-95107cbbeae5.mov

### Second method

Using the result from the *constrained* simulation, we can extract from `goSPL` outputs the required tectonic forcing to get a better fit with the paleo-elevation model.

This approach is explained in the notebook: `constrainedTec.ipynb` and consists in filtering the tectonic forcing for every 5 Ma period.


An example of input file for this type of approach is provided:

- `unconstrained.yml`

https://user-images.githubusercontent.com/7201912/141037144-2719a542-8cac-4cd1-91d4-fe64db7f4009.mov

### Third method

The third method does not required to run a *constrained* model first, and consist in iteratively (every 5Ma) forcing `goSPL` elevations based on the missmatch between model and observation.

This approach is explained in the notebook: `unconstrainedTec.ipynb` where we first explain how the result from `goSPL` could be statistically compared to paleo-elevation model and then refined by filtering the missmatch incrementally.


An example of input file for this type of approach is provided:

- `missmatch.yml`



