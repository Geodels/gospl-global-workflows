###
# ON Artemis hpc
# module load gcc/4.9.3 python/3.6.5 petsc-gcc-mpich/3.11.1
# pip install xarray netCDF4
###

import os
import argparse
import xarray as xr
import numpy as np
import pandas as pd
from mpi4py import MPI
from time import process_time

MPIcomm = MPI.COMM_WORLD
MPIrank = MPIcomm.Get_rank()
MPIsize = MPIcomm.Get_size()

# Parsing command line arguments
parser = argparse.ArgumentParser(
    description="This is a simple entry to extract outflow from goSPL model.",
    add_help=True,
)
parser.add_argument("-i", "--input", help="Input file name (csv file)", required=True)
parser.add_argument("-o", "--output", help="Output folder", required=True)
parser.add_argument(
    "-v",
    "--verbose",
    help="True/false option for verbose",
    required=False,
    action="store_true",
    default=True,
)

args = parser.parse_args()

df = pd.read_csv(args.input)

if MPIrank == 0:
    if not os.path.exists(args.output):
        os.makedirs(args.output)

for f in range(len(df)):

    t0 = process_time()

    step = df["time"].iloc[f].item()
    ncfile = df["netcdf"].iloc[f]

    if MPIrank == 0 and args.verbose:
        print("\nOpen output ", ncfile)

    dataset = xr.open_dataset(ncfile)
    basinNb = dataset.basinID.values.max()
    if f == 0:
        lats = dataset.latitude.values
        lons = dataset.longitude.values

    basins = dataset.basinID.values
    flow = dataset.flowDischarge.values
    sed = dataset.sedimentLoad.values

    if f == 0:
        mlon, mlat = np.meshgrid(lons, lats)
        posXY = np.dstack([mlon.ravel(), mlat.ravel()])[0]

    data = np.column_stack((posXY, basins.flatten()))
    data = np.column_stack((data, flow.flatten()))
    data = np.column_stack((data, sed.flatten()))

    basinNb = int(data[:, 2].max()) + 1

    count = int(basinNb / MPIsize)
    remainder = int(basinNb % MPIsize)
    start = MPIrank * count + min(MPIrank, remainder)
    stop = (MPIrank + 1) * count + min(MPIrank + 1, remainder)

    if MPIrank == 0 and args.verbose:
        print("  +  Nb of basins", basinNb - 1, " - approx. nb per CPU ", count)

    flowdata = np.zeros((basinNb, 3)) - 500.0
    seddata = np.zeros((basinNb, 3)) - 500.0

    p = 0
    for k in range(start, stop):

        if MPIrank == 0:
            if p % 500 == 0:
                print("    - it. ", p)

        ids = np.where(data[:, 2] == k)[0]
        if len(ids) > 10:
            tmp = data[ids, :]
            ifmax = np.where(tmp[:, 3] == tmp[:, 3].max())[0]
            if len(ifmax) > 0:
                flowdata[k, 0:2] = tmp[ifmax[0], 0:2]
                flowdata[k, -1] = tmp[ifmax[0], 3]

            ismax = np.where(tmp[:, 4] == tmp[:, 4].max())[0]
            if len(ismax) > 0:
                seddata[k, 0:2] = tmp[ismax[0], 0:2]
                seddata[k, -1] = tmp[ismax[0], 4]
        p += 1

    # Sediment discharge
    lon = seddata[:, 0].ravel()
    lat = seddata[:, 1].ravel()
    val = seddata[:, 2].ravel()
    MPIcomm.Allreduce(MPI.IN_PLACE, lon, op=MPI.MAX)
    MPIcomm.Allreduce(MPI.IN_PLACE, lat, op=MPI.MAX)
    MPIcomm.Allreduce(MPI.IN_PLACE, val, op=MPI.MAX)
    if MPIrank == 0:
        data = {
            "lon": lon,
            "lat": lat,
            "val": val,
        }
        df2 = pd.DataFrame(data)
        df2 = df2.drop(df2[df2.lon < -180].index)
        df2.to_csv(args.output + "/sed" + str(step) + ".csv", index_label="basin")

    # Flow discharge
    lon = flowdata[:, 0].ravel()
    lat = flowdata[:, 1].ravel()
    val = flowdata[:, 2].ravel()
    MPIcomm.Allreduce(MPI.IN_PLACE, lon, op=MPI.MAX)
    MPIcomm.Allreduce(MPI.IN_PLACE, lat, op=MPI.MAX)
    MPIcomm.Allreduce(MPI.IN_PLACE, val, op=MPI.MAX)
    if MPIrank == 0:
        data = {
            "lon": lon,
            "lat": lat,
            "val": val,
        }
        df2 = pd.DataFrame(data)
        df2 = df2.drop(df2[df2.lon < -180].index)
        df2.to_csv(args.output + "/flow" + str(step) + ".csv", index_label="basin")

    if MPIrank == 0 and args.verbose:
        print(" +  File execution took (%0.02f seconds)" % (process_time() - t0))
