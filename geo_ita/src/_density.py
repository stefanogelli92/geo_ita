from typing import Optional, Union, List

import numpy as np
import pandas as pd
import geopandas as gpd
from bokeh.io import show
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors

from geo_ita.src._data_enrichment import _get_margins, _create_geo_dataframe


class DensityCalculator:
    def __init__(
        self,
        df,
        latitude_column: Optional[str] = None,
        longitude_column: Optional[str] = None,
        geometry_column: Optional[str] = None,
        value_column: Optional[str] = None,
        filter_comune: Optional[Union[str, List[str]]] = None,
        filter_provincia: Optional[Union[str, List[str]]] = None,
        filter_regione: Optional[Union[str, List[str]]] = None,
        bandwidth_factor: float = 1.5,
        adaptive: bool = True,
    ):
        """
        Class for computing density using kernel methods, with optional weights.

        Args:
            df (pd.DataFrame): Input DataFrame containing points with latitude and longitude.
            latitude_column (str, optional): Column name for latitude. Automatically detected if not provided.
            longitude_column (str, optional): Column name for longitude. Automatically detected if not provided.
            geometry_column (str, optional): Column name for geometry. Automatically detected if not provided.
            value_column (str, optional): Column name for the value to use for density computation.
            filter_comune (str or list of str, optional): Filter data by Comune.
            filter_provincia (str or list of str, optional): Filter data by Provincia.
            filter_regione (str or list of str, optional): Filter data by Regione.
            bandwidth_factor (float): Scaling factor for bandwidth.
            adaptive (bool): Whether to use adaptive kernel bandwidth.
        """

        margins, shape = _get_margins(filter_comune=filter_comune,
                                      filter_provincia=filter_provincia,
                                      filter_regione=filter_regione)

        df = _create_geo_dataframe(df, lat_tag=latitude_column, long_tag=longitude_column, geo_tag=geometry_column)
        self.original_crs = df.crs
        df = df.to_crs({'init': 'epsg:3857'})
        if filter_regione or filter_comune or filter_provincia:
            df = df.to_crs({'init': 'epsg:3857'})
            df = gpd.tools.sjoin(df, shape, op='within')
        df = df.to_crs(self.original_crs)
        self.value_column = value_column
        self.bandwidth_factor = bandwidth_factor
        self.adaptive = adaptive
        self.kde = self._fit_kde(df)

    def _fit_kde(self, df):
        """
        Initialize the KDE with optional weights and adaptive bandwidth.

        Returns:
            gaussian_kde: KDE object.
        """
        points = np.array([(point.x, point.y) for point in df.geometry.centroid])
        nbrs = NearestNeighbors(n_neighbors=2).fit(points)
        distances, _ = nbrs.kneighbors(points)
        mean_distance = distances[:, 1].mean()
        self.mean_distance = mean_distance
        if self.adaptive:
            bandwidth = mean_distance * self.bandwidth_factor
        else:
            bandwidth = "scott"  # Use scipy's default rule of thumb

        weights = df[self.value_column].values if self.value_column else None

        return gaussian_kde(points.T, weights=weights, bw_method=bandwidth)

    def density_at(self, x, y):
        """
        Compute density at a specific point.

        Args:
            x, y (float): Coordinates of the point.

        Returns:
            float: Density value.
        """
        return self.kde(np.array([[x], [y]]))[0]


import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree, distance_matrix
import matplotlib.pyplot as plt


class SpatialDensity:
    """
    A class to calculate and visualize spatial density using:
    - Simple Density (weighted sum/count within a radius)
    - Kernel Density Estimation (KDE)
    - Inverse Distance Weighting (IDW)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        latitude_column: Optional[str] = None,
        longitude_column: Optional[str] = None,
        geometry_column: Optional[str] = None,
        value_column: Optional[str] = None,
        filter_comune: Optional[Union[str, List]] = None,
        filter_provincia: Optional[Union[str, List]] = None,
        filter_regione: Optional[Union[str, List]] = None,
        method: str = "kde",
        kde_bandwidth: Optional[Union[float, str]] = None,
        radius: Optional[float] = None,
        power: Optional[int] = 2,
    ):
        """
        Initialize the SpatialDensity class.

        Args:
            df (pd.DataFrame): Input DataFrame containing points with latitude and longitude.
            latitude_column (str, optional): Column name for latitude. Automatically detected if not provided.
            longitude_column (str, optional): Column name for longitude. Automatically detected if not provided.
            geometry_column (str, optional): Column name for geometry. Automatically detected if not provided.
            value_column (str, optional): Column name for the value to use for density computation.
            filter_comune (str or list of str, optional): Filter data by Comune.
            filter_provincia (str or list of str, optional): Filter data by Provincia.
            filter_regione (str or list of str, optional): Filter data by Regione.
            method (str): Method to use ('simple', 'kde', 'idw').
            kde_bandwidth (float): Bandwidth for KDE (only used in 'kde').
        """
        margins, shape = _get_margins(filter_comune=filter_comune,
                                      filter_provincia=filter_provincia,
                                      filter_regione=filter_regione)

        df = _create_geo_dataframe(df, lat_tag=latitude_column, long_tag=longitude_column, geo_tag=geometry_column)
        self.original_crs = df.crs
        df = df.to_crs({'init': 'epsg:3857'})
        if filter_regione or filter_comune or filter_provincia:
            df = df.to_crs({'init': 'epsg:3857'})
            df = gpd.tools.sjoin(df, shape, op='within')
        df = df.to_crs(self.original_crs)

        self.value_column = value_column
        self.method = method
        self.mean_distance = self._get_mean_distances(df["geometry"])

        if method.lower() == "kde":
            self._fit_kde(df, kde_bandwidth)
        elif method.lower() == "simple":
            self._fit_simple(df, radius)
        elif method.lower() == "idw":
            self._fit_idw(df, radius, power)
        else:
            raise ValueError("Method must be one of 'simple', 'kde', or 'idw'.")

    @staticmethod
    def _get_mean_distances(geo_series):
        """
        Calculate the mean distance between points.

        Args:
            df (pd.DataFrame): Input DataFrame containing points with latitude and longitude.

        Returns:
            float: Default radius value.
        """
        points = np.array([(point.x, point.y) for point in geo_series])
        nbrs = NearestNeighbors(n_neighbors=2).fit(points)
        distances, _ = nbrs.kneighbors(points)
        mean_distance = distances[:, 1].mean()
        return mean_distance

    def _fit_kde(self, df, bandwidth):
        """
        Initialize the KDE with optional weights and adaptive bandwidth.

        Returns:
            gaussian_kde: KDE object.
        """
        x = df["geometry"].x
        y = df["geometry"].y
        weights = df[self.value_column] if self.value_column else None
        self.kde_model = gaussian_kde(np.vstack([x, y]), weights=weights, bw_method=bandwidth)

    def _fit_simple(self, df, radius):
        """
        Initialize the KDTree for fast spatial queries.

        Returns:
            cKDTree: KDTree object.
        """
        self.points = df[["geometry"]].values
        self.radius = radius or self.mean_distance * 1.5
        self.weights = df[self.value_column].values if self.value_column else None
        self.tree = cKDTree(self.points)

    def _fit_idw(self, df, radius, power):
        """
        Initialize the IDW interpolation.

        Returns:
            np.array: Coordinates of the data points.
            np.array: Values of the data points.
        """
        self.coords = df[["geometry"]].values
        self.values = df[self.value_column].values
        self.radius = radius or self.mean_distance * 1.5
        self.power = power

    def calculate_density(
        self, points: Union[pd.DataFrame, np.array, tuple, list],
        latitude_column: Optional[str] = None,
        longitude_column: Optional[str] = None,
        geometry_column: Optional[str] = None,
    ):
        """
        Calculate density at specified points.

        Parameters:
        - points (pd.DataFrame, np.array, tuple, list): Points to calculate density for.
        - latitude_column (str, optional): Column name for latitude. Automatically detected if not provided.
        - longitude_column (str, optional): Column name for longitude. Automatically detected if not provided.
        - geometry_column (str, optional): Column name for geometry. Automatically detected if not provided.

        Returns:
        - np.array: Density values at specified points.
        """
        if isinstance(points, tuple):
            points = pd.DataFrame([{"longitude": points[0], "latitude": points[1]}])
        elif isinstance(points, list):
            points = pd.DataFrame(points, columns=["longitude", "latitude"])

        points = _create_geo_dataframe(points, lat_tag=latitude_column, long_tag=longitude_column, geo_tag=geometry_column)

        if self.method == "simple":
            return self._calculate_simple_density(points)
        elif self.method == "kde":
            return self._calculate_kde_density(points)
        elif self.method == "idw":
            return self._calculate_idw_density(points)

    def _calculate_simple_density(self, df):
        """
        Calculate density at specified points using simple density method.

        Parameters:
        - points (pd.DataFrame): DataFrame with 'latitude' and 'longitude'.

        Returns:
        - np.array: Density values at specified points.
        """
        target_coords = df[["geometry"]].values
        indices = self.tree.query_ball_point(target_coords, self.radius)
        densities = [
            sum(self.weights[idx]) if idx and self.value_column else len(idx)
            for idx in indices
        ]
        return np.array(densities)

    def _calculate_kde_density(self, df):
        """
        Calculate density at specified points using KDE method.

        Parameters:
        - df (pd.DataFrame): DataFrame with 'latitude' and 'longitude'.

        Returns:
        - np.array: Density values at specified points.
        """
        coords = np.vstack([df["geometry"].x, df["geometry"].y])
        return self.kde_model(coords)

    def _calculate_idw_density(self, df):
        """
        Calculate density at specified points using IDW method.

        Parameters:
        - df (pd.DataFrame): DataFrame with 'latitude' and 'longitude'.

        Returns:
        - np.array: Density values at specified points.
        """
        target_coords = df[["geometry"]].values
        distances = distance_matrix(target_coords, self.coords)
        weights = 1 / (distances ** self.power)
        weights[distances == 0] = np.inf
        return (weights @ self.values) / np.sum(weights, axis=1)

    def plot(
        self,
        filter_comune: Optional[Union[str, List]] = None,
        filter_provincia: Optional[Union[str, List]] = None,
        filter_regione: Optional[Union[str, List]] = None,
        grid_size: Optional[float] = None,
        interactive: bool = False,
        save_in_path: Optional[str] = None,
        show_flag: bool = True,
    ):
        """
        Plot the interpolated density over a grid.

        Parameters:
            - filter_comune (str or list of str, optional): Filter data by Comune.
            - filter_provincia (str or list of str, optional): Filter data by Provincia.
            - filter_regione (str or list of str, optional): Filter data by Regione.
            - grid_size (int, optional): Resolution of the grid.
        """
        margins, shape = _get_margins(filter_comune=filter_comune,
                                      filter_provincia=filter_provincia,
                                      filter_regione=filter_regione)

        shape = shape.dissolve(by='key')

        X, Y, Z = self.__interpolate_grid(margins, shape, grid_size)

        if interactive:
            self.__plot_interactive(X, Y, Z, shape, save_in_path, show_flag)
        else:
            self.__plot_static(X, Y, Z)

    def __plot_interactive(self, X, Y, Z, shape, save_in_path, show_flag):
        """
        Internal method to plot an interactive density map with bokeh.
        """
        from bokeh.plotting import figure, save
        from bokeh.models import ColumnDataSource, HoverTool, GeoJSONDataSource
        from bokeh.palettes import Viridis256
        from bokeh.transform import linear_cmap
        from bokeh.io import output_notebook

        output_notebook()

        # Create the plot
        p = figure(title=f"Spatial Density ({self.method.upper()})", plot_width=800, plot_height=600)
        p.xaxis.axis_label = "Longitude"
        p.yaxis.axis_label = "Latitude"

        # Create the color mapper
        color_mapper = linear_cmap(field_name='Z', palette=Viridis256, low=min(Z), high=max(Z))

        # Add the density map
        p.patches('X', 'Y', source=ColumnDataSource(data=dict(X=X, Y=Y, Z=Z)),
                  fill_color=color_mapper, line_color=None, fill_alpha=0.7)

        # Add the shape
        geo_source = GeoJSONDataSource(geojson=shape.to_json())
        p.patches('xs', 'ys', source=geo_source, fill_color=None, line_color="black", line_width=2)

        # Add the hover tool
        hover = HoverTool()
        hover.tooltips = [("Latitude", "@Y"), ("Longitude", "@X"), ("Density", "@Z")]
        p.add_tools(hover)

        # Save or show the plot
        if save_in_path:
            save(p, filename=save_in_path)
        if show_flag:
            show(p)

    def __plot_static(self, X, Y, Z):
        """
        Internal method to plot a static density map.
        """
        plt.figure(figsize=(10, 6))
        plt.contourf(X, Y, Z, cmap="viridis", levels=15)
        plt.colorbar(label="Density")
        plt.title(f"Spatial Density ({self.method.upper()})")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.show()

    def __interpolate_grid(self, margins, shape, grid_size):
        """
        Internal method to interpolate density over a grid within a shape.

        Parameters:
        - margins (tuple): Margins of the shape.
        - shape (shapely.geometry): Shape to interpolate within.
        - grid_size (int): Resolution of the grid.

        Returns:
        - X, Y, Z: Grid coordinates and interpolated density values.
        """
        xmin, ymin, xmax, ymax = margins[0][0], margins[0][1], margins[1][0], margins[1][1]

        grid_size = grid_size or self.mean_distance * 0.5
        X, Y = np.meshgrid(np.arange(xmin, xmax, grid_size), np.arange(ymin, ymax, grid_size))
        grid_points = pd.DataFrame({
            "longitude": X.ravel(),
            "latitude": Y.ravel()
        })
        grid_points = _create_geo_dataframe(grid_points, lat_tag="latitude", long_tag="longitude")

        # Filter points within the shape
        grid_points = gpd.sjoin(grid_points, shape, op='within')

        Z = self.calculate_density(grid_points)
        X = grid_points.geometry.x.values
        Y = grid_points.geometry.y.values
        return X, Y, Z.reshape(X.shape)
