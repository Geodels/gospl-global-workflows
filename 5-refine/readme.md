# Temporal refinement of goSPL model for bio-like model

Often paleo-elevation reconstruction and associated plate model are provided for coarse time intervals (1 to 5 Ma increments). However in some cases, like when one want to run a ecological model, these time steps are too large and it is required to refine the temporal evolution.

+ One solution consists in outputing `goSPL` results at a finer temporal level but this approach (specially if you which to constrain your model over time with available paleo-elevation dataset) will rely on assimilating data at finer temporal scale to limit errors propagation over time.
+ Another consists in running the `goSPL` model at coarse temporal resolution and then interpolate the output to a finer temporal scale using the plate-boundary velocities.

This notebook provides a method for these 2 solution and allows to build a temporally refined paleo-elevation reconstruction based on a coarser one.

The method combines plate movements, clustering techniques and incremental interpolations using a backward/forward approach and has been designed to increase the temporal resolution by a factor of 5, e.g. from a 5 Ma to a 1 Ma time step. It can easily been modified to increase this factor or could be ran multiple times to reach the desired time interval.

<img width="754" alt="example 1Ma dt elevation" src="https://user-images.githubusercontent.com/7201912/144146410-2d7eb2ed-16fb-4e97-9591-7094603c5bbb.png">

<img width="754" alt="example 1Ma dt elevation" src="https://user-images.githubusercontent.com/7201912/144146416-3c8b81ce-6388-4dd0-b129-c90f058339f5.png">
