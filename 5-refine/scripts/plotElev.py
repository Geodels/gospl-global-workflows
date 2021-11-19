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


def getBounds(time, topology_features, rotation_model):
    resolved_topologies = []
    shared_boundary_sections = []
    pygplates.resolve_topologies(
        topology_features,
        rotation_model,
        resolved_topologies,
        time,
        shared_boundary_sections,
    )
    wrapper = pygplates.DateLineWrapper(0.0)
    subductions = []
    oceanRidges = []
    otherBounds = []
    for shared_boundary_section in shared_boundary_sections:
        if shared_boundary_section.get_feature().get_feature_type() == pygplates.FeatureType.create_gpml(
            "MidOceanRidge"
        ):
            for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():
                split_geometry = wrapper.wrap(shared_sub_segment.get_geometry())
                for geometry in split_geometry:
                    X = []
                    Y = []
                    for point in geometry.get_points():
                        X.append(point.get_longitude()), Y.append(point.get_latitude())
                    x, y = X, Y
                    subductions.append([x, y])
        elif shared_boundary_section.get_feature().get_feature_type() == pygplates.FeatureType.create_gpml(
            "SubductionZone"
        ):
            for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():
                split_geometry = wrapper.wrap(shared_sub_segment.get_geometry())
                for geometry in split_geometry:
                    X = []
                    Y = []
                    for point in geometry.get_points():
                        X.append(point.get_longitude()), Y.append(point.get_latitude())
                    x, y = X, Y
                    oceanRidges.append([x, y])
        else:
            for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():
                split_geometry = wrapper.wrap(shared_sub_segment.get_geometry())
                for geometry in split_geometry:
                    X = []
                    Y = []
                    for point in geometry.get_points():
                        X.append(point.get_longitude()), Y.append(point.get_latitude())
                    x, y = X, Y
                    otherBounds.append([x, y])
    return subductions, oceanRidges, otherBounds


def plotElev(time, data, subductions, oceanRidges, otherBounds, out_path, show=False):
    fig = pygmt.Figure()
    with pygmt.config(FONT="6p,Helvetica,black"):
        pygmt.makecpt(cmap="geo", series=[-6000, 6000])
        fig.basemap(region="d", projection="W6i", frame="afg")
        viewset = data.z
        fig.grdimage(viewset, shading=True, frame=False)
        #         fig.grdcontour(interval=0.1,grid=viewset,limit=[-0.1, 0.1])
        fig.colorbar(
            position="jBC+o0c/-1.5c+w8c/0.3c+h", frame=["a2000", "x+lElevation", "y+lm"]
        )

        for k in range(len(subductions)):
            fig.plot(
                x=subductions[k][0], y=subductions[k][1], pen="1p,red", transparency="0"
            )

        for k in range(len(oceanRidges)):
            fig.plot(
                x=oceanRidges[k][0],
                y=oceanRidges[k][1],
                pen="1p,white",
                transparency="0",
            )

        for k in range(len(otherBounds)):
            fig.plot(
                x=otherBounds[k][0],
                y=otherBounds[k][1],
                pen="1p,purple",
                transparency="0",
            )

    # Customising the font style
    fig.text(text=str(time) + " Ma", position="TL", font="8p,Helvetica-Bold,black")
    fname = out_path + "/elev" + str(time) + "Ma.png"
    fig.savefig(fname=fname, dpi=500)
    if show:
        fig.show(dpi=500, width=1000)

    return


def viewElev(time, topoFeature, rotationModel, out_path, show=False):

    subductions, oceanRidges, otherBounds = getBounds(
        int(time), topoFeature, rotationModel
    )
    elevfile = out_path + "/ndem" + str(time) + "Ma.nc"
    data = xr.open_dataset(elevfile)
    plotElev(time, data, subductions, oceanRidges, otherBounds, out_path, show)

    return
