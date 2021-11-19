import os
import pygmt
import struct
import imageio
import pygplates
import subprocess
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import cartopy.crs as ccrs
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as scpRot


class refineTime:
    def __init__(
        self,
        rotation=None,
        plate=None,
        dem=None,
        vel=None,
        out=None,
        rain=None,
        nproc=4,
        clustngbh=6,
        clustdist=10.0e3,
    ):

        self.fdem = dem
        self.fvel = vel
        self.frain = rain

        # Define output folder name for the simulation
        if not os.path.exists(out):
            os.makedirs(out)
        self.forwpath = out
        self.backpath = out + "_back"
        if not os.path.exists(self.backpath):
            os.makedirs(self.backpath)

        self.radius = 6371 * 1000.0

        self.rotationModel = pygplates.RotationModel(rotation)
        self.topoFeature = pygplates.FeatureCollection(plate)

        self.glon = None
        self.glat = None
        self.lonlat = None
        self.sXYZ = None
        self.vtree = None

        self.nprocs = nproc
        self.clustngbh = clustngbh
        self.clustdist = clustdist

        self.val = [0.2, 0.4, 0.6, 0.8]

        return

    def runInterval(self, time, dt, forward=True, gfilt=0.75):

        paleoZ = self.getPaleoTopo(time[0])

        out_path = self.backpath
        if forward:
            out_path = self.forwpath
            out_path2 = self.backpath

        if self.glon is None:
            self.glon = paleoZ.longitude.values
            self.glat = paleoZ.latitude.values
            self.shape = paleoZ.z.shape
            lons, lats = np.meshgrid(self.glon, self.glat)
            self.lonlat = np.empty((len(lons.ravel()), 2))
            self.lonlat[:, 0] = lons.ravel()
            self.lonlat[:, 1] = lats.ravel()

        if self.frain is not None:
            paleoR = self.getPaleoRain(time[0])
            rdata = []
            rdata.append(paleoR)
            newrdata = paleoR.copy()
            newrdata = newrdata.drop_vars(names=["z"])
            newrdata["z"] = (["lat", "lon"], paleoR.z.values)
            newrdata.to_netcdf(out_path + "/nrain" + str(time[0]) + "Ma.nc")

        ndata = []
        ndata.append(paleoZ)
        nheights = gaussian_filter(paleoZ.z.values, sigma=gfilt)
        newdata = paleoZ.copy()
        newdata = newdata.drop_vars(names=["z"])
        newdata["z"] = (["latitude", "longitude"], nheights)
        newdata.to_netcdf(out_path + "/ndem" + str(time[0]) + "Ma.nc")

        print(" + start loop", time[0])
        for s in range(len(time) - 1):

            plateIds = self.getPlateIDs(time[s])
            heights = ndata[s].z.values.ravel()
            sphericalZ = heights + self.radius
            if self.sXYZ is None:
                self.sXYZ = self.polarToCartesian(
                    sphericalZ, self.lonlat[:, 0], self.lonlat[:, 1]
                )
            rotations = self.getRotations(time[s], dt, plateIds)
            movXYZ = self.movePlates(plateIds, rotations)

            nheights = self.clusterZ(time[s], movXYZ, heights)
            newZ = self.interpData(nheights, movXYZ).reshape(self.shape)

            if self.frain is not None:
                newR = self.interpData(rdata[s].z.values.ravel(), movXYZ).reshape(
                    self.shape
                )

            if forward:
                bdata = xr.open_dataset(
                    out_path2 + "/ndem" + str(time[s] - dt) + "Ma.nc"
                )
                diff = (bdata.z.values - newZ) * self.val[s]
                nheights = gaussian_filter(newZ + diff, sigma=gfilt)

            data = ndata[s].copy()
            data = data.drop_vars(names=["z"])
            if forward:
                data["z"] = (["latitude", "longitude"], nheights)
            else:
                data["z"] = (["latitude", "longitude"], newZ)
            data.to_netcdf(out_path + "/ndem" + str(time[s] - dt) + "Ma.nc")
            ndata.append(data)

            if self.frain is not None:
                data = rdata[s].copy()
                data = data.drop_vars(names=["z"])
                data["z"] = (["lat", "lon"], newR)
                data.to_netcdf(out_path + "/nrain" + str(time[s] - dt) + "Ma.nc")
                rdata.append(data)

            print("    -  done time: ", time[s] - dt)

        return

    def refinePaleoData(self, times):
        ldt = times[0] - times[1]
        if ldt < 0:
            times = np.flip(times)
            ldt = -ldt
        dt = int(ldt / 5.0)

        for k in range(len(times) - 1):
            # Backward
            print("+ Backward run")
            time = np.arange(times[k + 1], times[k], dt)
            self.runInterval(time, -dt, forward=False)
            # Forward
            print("+ Forward run")
            time = np.arange(times[k], times[k + 1], -dt)
            self.runInterval(time, dt)

        args = [
            "cp",
            self.backpath + "/ndem" + str(int(times[-1])) + "Ma.nc",
            self.forwpath,
        ]
        self.runSubProcess(args, True, ".")

        return

    def getPaleoTopo(self, time):
        # Get the paleosurface mesh file (as netcdf file)
        paleoDemsPath = Path(self.fdem)
        initialLandscapePath = list(paleoDemsPath.glob("**/%dMa.nc" % int(time)))[0]
        # Open it with xarray
        data = xr.open_dataset(initialLandscapePath)
        return data.sortby(data.latitude)

    def getPaleoRain(self, rain_folder, time, glon, glat):
        # Get the paleosurface mesh file (as netcdf file)
        paleoRainPath = Path(self.frain)
        initialRainPath = list(paleoRainPath.glob("**/%dMa.nc" % int(time)))[0]
        # Open it with xarray
        data = xr.open_dataset(initialRainPath)
        datai = data.interp(lat=self.glat, lon=self.glon)
        return datai.sortby(data.lat)

    def getPlateIDs(self, time):
        # Read plate IDs from gPlates exports
        velfile = self.fvel + "/vel" + str(int(time)) + "Ma.xy"
        data = pd.read_csv(
            velfile,
            sep=r"\s+",
            engine="c",
            header=None,
            na_filter=False,
            dtype=float,
            low_memory=False,
        )
        data = data.drop_duplicates().reset_index(drop=True)
        gplateID = data.iloc[:, -1].to_numpy().astype(int)
        if self.vtree is None:
            llvel = data.iloc[:, 0:2].to_numpy()
            self.vtree = cKDTree(llvel)
        dist, ids = self.vtree.query(self.lonlat, k=1)

        return gplateID[ids]

    def polarToCartesian(self, radius, theta, phi, useLonLat=True):
        if useLonLat:
            theta, phi = np.radians(theta + 180.0), np.radians(90.0 - phi)
        X = radius * np.cos(theta) * np.sin(phi)
        Y = radius * np.sin(theta) * np.sin(phi)
        Z = radius * np.cos(phi)

        # Return data either as a list of XYZ coordinates or as a single XYZ coordinate
        if type(X) == np.ndarray:
            return np.stack((X, Y, Z), axis=1)
        else:
            return np.array([X, Y, Z])

    def cartesianToPolarCoords(self, XYZ, useLonLat=True):
        X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
        R = (X ** 2 + Y ** 2 + Z ** 2) ** 0.5
        theta = np.arctan2(Y, X)
        phi = np.arccos(Z / R)

        # Return results either in spherical polar or leave it in radians
        if useLonLat:
            theta, phi = np.degrees(theta), np.degrees(phi)
            lon, lat = theta - 180, 90 - phi
            lon[lon < -180] = lon[lon < -180] + 360
            return R, lon, lat
        else:
            return R, theta, phi

    def quaternion(self, axis, angle):
        return [
            np.sin(angle / 2) * axis[0],
            np.sin(angle / 2) * axis[1],
            np.sin(angle / 2) * axis[2],
            np.cos(angle / 2),
        ]

    def getRotations(self, time, deltaTime, plateIds):
        rotations = {}
        for plateId in np.unique(plateIds):
            stageRotation = self.rotationModel.get_rotation(
                int(time - deltaTime), int(plateId), int(time)
            )
            stageRotation = stageRotation.get_euler_pole_and_angle()
            axisLatLon = stageRotation[0].to_lat_lon()
            axis = self.polarToCartesian(1, axisLatLon[1], axisLatLon[0])
            angle = stageRotation[1]
            rotations[plateId] = scpRot.from_quat(self.quaternion(axis, angle))
        return rotations

    def movePlates(self, plateIds, rotations):
        newXYZ = np.copy(self.sXYZ)
        for idx in np.unique(plateIds):
            rot = rotations[idx]
            newXYZ[plateIds == idx] = rot.apply(newXYZ[plateIds == idx])
        return newXYZ

    def interpData(self, data, mvxyz, ngbh=1):
        # Build the kdtree
        ptree = cKDTree(mvxyz)
        distNbghs, idNbghs = ptree.query(self.sXYZ, k=ngbh)
        if ngbh == 1:
            return data[idNbghs]

        # Inverse weighting distance...
        weights = np.divide(
            1.0, distNbghs, out=np.zeros_like(distNbghs), where=distNbghs != 0,
        )
        onIDs = np.where(distNbghs[:, 0] == 0)[0]
        temp = np.sum(weights, axis=1)
        tmp = np.sum(weights * data[idNbghs], axis=1)
        # Elevation
        interpZ = np.divide(tmp, temp, out=np.zeros_like(temp), where=temp != 0)
        if len(onIDs) > 0:
            interpZ[onIDs] = data[idNbghs[onIDs, 0]]
        return interpZ

    def runSubProcess(self, args, output=True, cwd="."):
        p = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        lines = []
        while True:
            line = p.stdout.readline()
            if not line and p.poll() is not None:
                break
            lines.append(line)
            if output:
                print(line, end="")

        if p.returncode != 0:
            output = "".join(lines)
            if "ERROR: " in output:
                _, _, error_msg = output.partition("ERROR: ")
            elif "what()" in output:
                _, _, error_msg = output.partition("what(): ")
            else:
                error_msg = "dbscan aborted unexpectedly."
            error_msg = " ".join(error_msg.split())

            raise RuntimeError(error_msg)

    def clusterZ(
        self, time, mvxyz, elev, output=False, cwd=".",
    ):
        if output:
            print("\ndbscan MPI")
        dims = [len(mvxyz), 3]
        linepts = mvxyz.ravel()
        lgth = len(linepts)
        fbin = "nodes" + str(time) + ".bin"
        with open(fbin, mode="wb") as f:
            f.write(struct.pack("i" * 2, *[int(i) for i in dims]))
            f.write(struct.pack("f" * (lgth), *[float(i) for i in linepts]))
        fnc = "clusters" + str(time) + ".nc"
        mpi_args = [
            "mpirun",
            "-np",
            str(self.nprocs),
            "dbscan",
            "-i",
            fbin,
            "-b",
            "-m",
            "2",
            "-e",
            str(self.clustdist),
            "-o",
            fnc,
        ]
        self.runSubProcess(mpi_args, output, cwd)
        if output:
            print("\nGet global ID of clustered vertices")
        cluster = xr.open_dataset(fnc)
        isClust = cluster.cluster_id.values > 0
        clustPtsX = cluster.position_col_X0.values[isClust]
        clustPtsY = cluster.position_col_X1.values[isClust]
        clustPtsZ = cluster.position_col_X2.values[isClust]
        clustPts = np.vstack((clustPtsX, clustPtsY))
        clustPts = np.vstack((clustPts, clustPtsZ)).T
        ptree = cKDTree(mvxyz)
        dist, ids = ptree.query(clustPts, k=1)
        isCluster = np.zeros(len(mvxyz), dtype=int)
        isCluster[ids] = 1
        idCluster = isCluster > 0
        ptsCluster = mvxyz[idCluster]
        ctree = cKDTree(ptsCluster)
        _, clustNgbhs = ctree.query(ptsCluster, k=self.clustngbh)
        clustNgbhs = clustNgbhs[:, 1:]
        args = [
            "rm",
            fbin,
            fnc,
        ]
        self.runSubProcess(args, output, cwd)

        # Get heights of nearest neighbours
        heightsInCluster = elev[idCluster]
        neighbourHeights = heightsInCluster[clustNgbhs]

        # For points in cluster, set new heights to the maximum height of
        # nearest neighbours
        clustZ = elev.copy()
        neighbourHeights.partition(1, axis=1)
        clustZ[idCluster] = np.mean(
            neighbourHeights[:, -int(self.clustngbh / 2) :], axis=1
        )

        return clustZ

    def runSubProcess(self, args, output=True, cwd="."):
        # Launch a subprocess
        p = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        # Capture and re-print OpenMC output in real-time
        lines = []
        while True:
            # If OpenMC is finished, break loop
            line = p.stdout.readline()
            if not line and p.poll() is not None:
                break

            lines.append(line)
            if output:
                # If user requested output, print to screen
                print(line, end="")

        # Raise an exception if return status is non-zero
        if p.returncode != 0:
            # Get error message from output and simplify whitespace
            output = "".join(lines)
            if "ERROR: " in output:
                _, _, error_msg = output.partition("ERROR: ")
            elif "what()" in output:
                _, _, error_msg = output.partition("what(): ")
            else:
                error_msg = "dbscan aborted unexpectedly."
            error_msg = " ".join(error_msg.split())

            raise RuntimeError(error_msg)
