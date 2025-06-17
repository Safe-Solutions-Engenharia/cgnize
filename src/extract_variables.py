import numpy as np
import pandas as pd
from numba import njit
import h5py


def cgns_variable_list(h5py_file: h5py.File) -> list[str]:
    """
    Analyzes a CGNS/HDF5 file and returns a unique list with the names of
    all result variables found.

    This function is optimized to fetch only the names (HDF5 keys),
    without loading the numerical data.

    Args:
        h5py_file (h5py.File): The file object opened with h5py.File().

    Returns:
        list[str]: A list of strings, where each string is the name of a variable
                   available in the result data (e.g., 'Pressure', 'Velocity_X').
                   Returns an empty list if the expected path is not found.
    """
    variable_name: set[str] = set()
    zone_path = 'Results3D/Level#01_Zone#0000000001'

    try:
        zone_group = h5py_file[zone_path]
        
        for solution_group_name in zone_group:
            solution_group = zone_group[solution_group_name]
            
            if isinstance(solution_group, h5py.Group):
                for var_name in solution_group.keys():
                    variable_name.add(var_name)

    except KeyError:
        print(f"Warning: The default path '{zone_path}' was not found in the HDF5 file.")
        return []

    variable_name.difference_update({' data', ' link', ' path'}) 
    
    return sorted(list(variable_name))


def extract_cgns_variable(h5py_file: h5py.File, variable_name: str) -> np.ndarray:
    """
    Extracts and time-averages a variable from the last few solution steps
    in a CGNS/HDF5 file.

    This function identifies the last 'time_average_count' solution groups,
    extracts the data for the specified variable, removes a one-cell
    boundary layer, computes the mean over the time steps, and returns a
    flattened array.

    Args:
        h5py_file (h5py.File): The opened h5py file object.
        variable_name (str): The name of the variable to extract (e.g., 'Pressure').
        time_average_count (int): The number of recent time steps to average.
                                  Defaults to 3.

    Returns:
        np.ndarray: A 1D NumPy array with the time-averaged and flattened data.
    """
    zone_path = 'Results3D/Level#01_Zone#0000000001'
    
    try:
        zone_group = h5py_file[zone_path]
        
        valid_keys: list[str] = []
        for key, value in zone_group.items():
            if isinstance(value, h5py.Group):
                if variable_name in value:
                    valid_keys.append(key)

        sorted_keys = sorted(valid_keys)
        
        data_snapshots = [
            zone_group[f"{key}/{variable_name}/ data"][1:-1, 1:-1, 1:-1]
            for key in sorted_keys
        ]

        #TODO: modulate better for more options!
        try:
            final_array = np.array(data_snapshots)
        except ValueError:
            shapes = [d.shape for d in data_snapshots]
            raise ValueError(f"Could not create NumPy array. The datasets have inconsistent shapes: {shapes}")
        
        final_array = final_array[-3:, ...]
        final_array = np.mean(final_array, axis=0)
        final_array = final_array.reshape(-1)

        return final_array
            
    except (KeyError, AttributeError, TypeError) as e:
        print(f"File structure or type error: Verify the path and format of the CGNS file. Details: {e}")
        raise


@njit
def _transform_coordinates(grid_x: np.ndarray, grid_y: np.ndarray, 
                           grid_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This is an internal function.The get_coordinates() function uses it to transform the structure
    that h5py returns by default into a structure ready to be aligned
    with a more common one.

    Args:
        The coordinates of the points.

    Returns:
        The corrected coordinates.
    """

    # X Coordinates
    grid_x = grid_x[1:, 1:, :]
    x_grid2 = (grid_x[:, :, :-1] + grid_x[:, :, 1:]) / 2
    x_flat = np.ascontiguousarray(x_grid2).reshape(-1)

    # Y Coordinates
    y_grid2 = (grid_y[:, :-1, :] + grid_y[:, 1:, :]) / 2
    y_flat = np.ascontiguousarray(y_grid2[:-1, :, 1:]).reshape(-1)

    # Z Coordinates
    z_grid = (grid_z[:-1, :-1, 1:] + grid_z[1:, :-1, 1:]) / 2
    z_flat = np.ascontiguousarray(z_grid).reshape(-1)

    return x_flat, y_flat, z_flat


def get_coordinates(cgns_file: h5py) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function extracts coordinates from the .cgns
    file using h5py; however, that library returns the points in a somewhat
    unconventional format. Therefore, this code also transforms the data
    and returns it in a more convenient format.

    Args:
        cgns_file (h5py.File): The h5py file object to read from.

    Returns:
        A tuple containing the corrected X, Y, and Z coordinates.
    """

    try:
        grid_x = cgns_file['Results3D/Level#01_Zone#0000000001/GridCoordinates/ link/CoordinateX/ data']
        grid_y = cgns_file['Results3D/Level#01_Zone#0000000001/GridCoordinates/ link/CoordinateY/ data']
        grid_z = cgns_file['Results3D/Level#01_Zone#0000000001/GridCoordinates/ link/CoordinateZ/ data']

    except Exception:
        print(f"Error reading coordinates from file: {cgns_file}")
        return
    
    grid_x = np.array(grid_x)
    grid_y = np.array(grid_y)
    grid_z = np.array(grid_z)

    x_grid, y_grid, z_grid = _transform_coordinates(grid_x, grid_y, grid_z)

    return x_grid, y_grid, z_grid


def generate_cgns_df(cgns_path: str, variables: list[str]) -> pd.DataFrame:
    """
    This function aims to create an easy-to-use DataFrame with information
    from the read .cgns file. With this DataFrame, it is much easier to
    assign an ID to each parallelogram in the grid and to find the maximum
    and minimum values of our coordinates.

    Args:
        cgns_path (str): The file path for the .cgns file.
        variables (list[str]): A list of variable names to extract from the file.

    Returns:
        pd.DataFrame: A DataFrame with the X, Y, and Z information for all
        parallelograms, in addition to the data for each specified variable.
    """
    cgns_file = h5py.File(cgns_path, "r")

    x, y, z = get_coordinates(cgns_file)

    df_cgns = pd.DataFrame()
    df_cgns["X"] = x
    df_cgns["Y"] = y
    df_cgns["Z"] = z
    for var in variables:
        var_array = extract_cgns_variable(cgns_file, var)
        df_cgns[var] = var_array

    return df_cgns
