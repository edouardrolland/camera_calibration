import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import pymap3d
import folium
import os

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the haversine distance between two points in decimal degrees.

    Parameters:
        lat1, lon1 : Latitude and longitude of the first point.
        lat2, lon2 : Latitude and longitude of the second point.

    Returns:
        Distance in meters.
    """
    R_earth = 6371000  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R_earth * c
    return distance

def getObjectPosition_flatPixel(x_image, y_image, f, cx, cy, gb_pitch, gb_yaw, lat_drone, lon_drone, alt_drone):
    # Position of the object in the image expressed in the camera frame
    r_c = np.array([[f],[x_image - cx],[y_image - cy]])

    # Rotation matrix from camera frame to world frame
    C_cw = R.from_euler('ZY', [gb_yaw, gb_pitch], degrees=True).as_matrix()

    # Computing the intersection with the ground
    alpha = alt_drone/np.matmul(C_cw,r_c)[2][0]
    r_w = alpha*np.matmul(C_cw,r_c) + np.array([[0], [0], [-alt_drone]])
    
    # Compute position in earth frame
    animalPosition = pymap3d.ned2geodetic(r_w[0][0], r_w[1][0], 0, lat_drone, lon_drone, 0)[0:2]

    results = {"Lat":animalPosition[0], "Lng":animalPosition[1]}
    return results

def main():
    # Replace 'your_file.csv' with the actual path to your CSV file
    csv_file = '/home/edr/workspace/wildview/camera_calibration/Multigimbal experiments/output.csv'

    if not os.path.exists(csv_file):
        print(f"The file '{csv_file}' was not found.")
        return

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Check required columns
    required_columns = ['start_x', 'start_y', 'f_mini3', 'cx', 'cy', 'gb_pitch', 'gb_yaw', 'lat_drone', 'lon_drone', 'altitude_drone', 'annotation_x', 'annotation_z']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The column '{col}' is missing from the CSV file.")

    # Initial guesses for f, cx, cy
    initial_f = df['f_mini3'].mean()
    initial_cx = df['cx'].mean()
    initial_cy = df['cy'].mean()
    initial_guess = [initial_f, initial_cx, initial_cy]
    print(f"Initial estimates:\nf = {initial_f}\ncx = {initial_cx}\ncy = {initial_cy}")

    # Define the loss function (sum of squared distances)
    def loss(params):
        f, cx, cy = params
        total_error = 0.0
        for idx, row in df.iterrows():
            predicted_pos = getObjectPosition_flatPixel(
                row['start_x'], row['start_y'], f, cx, cy,
                row['gb_pitch'], row['gb_yaw'],
                row['lat_drone'], row['lon_drone'], row['altitude_drone']
            )

            actual_lat = 49.306799
            actual_lon = 4.593616
         
            distance = haversine_distance(predicted_pos['Lat'], predicted_pos['Lng'], actual_lat, actual_lon)
            total_error += distance ** 2  # sum of squared distances
        return total_error

    # Perform the optimization
    result = minimize(loss, initial_guess, method='L-BFGS-B')

    if result.success:
        optimized_f, optimized_cx, optimized_cy = result.x
        print(f"Optimized parameters:\nf = {optimized_f}\ncx = {optimized_cx}\ncy = {optimized_cy}")
    else:
        print("Optimization failed:", result.message)
        return

    # Compute new waypoints with optimized parameters
    new_waypoints = []
    for idx, row in df.iterrows():
        predicted_pos = getObjectPosition_flatPixel(
            row['start_x'], row['start_y'], optimized_f, optimized_cx, optimized_cy,
            row['gb_pitch'], row['gb_yaw'],
            row['lat_drone'], row['lon_drone'], row['altitude_drone']
        )
        new_waypoints.append(predicted_pos)

    # Add new waypoints to DataFrame
    df['optimized_annotation_x'] = [wp['Lat'] for wp in new_waypoints]
    df['optimized_annotation_z'] = [wp['Lng'] for wp in new_waypoints]

    actual_lat = 49.306799
    actual_lon = 4.593616

    # Compute distances between optimized waypoints and actual annotations
    df['distance_m'] = df.apply(
        lambda row: haversine_distance(row['optimized_annotation_x'], row['optimized_annotation_z'], 49.306799, 4.593616) if not pd.isnull(row['optimized_annotation_x']) else np.nan,
        axis=1
    )

    # Compute mean error (ignoring NaN)
    average_error = df['distance_m'].mean()
    print(f"Average error after optimization: {average_error:.2f} meters")

    # Save optimized waypoints to new CSV file
    output_csv = 'optimized_waypoints.csv'
    df.to_csv(output_csv, index=False)
    print(f"Optimized waypoints saved in '{output_csv}'")

    # Create an interactive map with Folium
    # Compute map center
    center_lat = df[['annotation_x', 'optimized_annotation_x']].mean().mean()
    center_lon = df[['annotation_z', 'optimized_annotation_z']].mean().mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    # Add original annotations
    for idx, row in df.iterrows():
        folium.Marker(
            location=[row['annotation_x'], row['annotation_z']],
            popup=f"Original Annotation {idx+1}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    # Add optimized annotations
    for idx, row in df.iterrows():
        folium.Marker(
            location=[row['optimized_annotation_x'], row['optimized_annotation_z']],
            popup=f"Optimized Annotation {idx+1}<br>Error: {row['distance_m']:.2f} m",
            icon=folium.Icon(color='green', icon='info-sign')
        ).add_to(m)

    # Save the map to HTML file
    map_file = 'optimized_waypoints_map.html'
    m.save(map_file)
    print(f"Map with optimized waypoints saved in '{map_file}'")

    # Optional: Show distances and average error
    print("\nDistances of each optimized annotation compared to the original annotation (in meters):")
    for idx, distance in enumerate(df['distance_m'], start=1):
        print(f"Annotation {idx}: {distance:.2f} meters")

    print(f"\nAverage error (mean distance): {average_error:.2f} meters")
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Filter only values where gb_pitch is -89.9 or -90.0
    df_filtered = df[df['gb_pitch'].isin([-89.9, -90.0])]

    # Add grouped altitude column
    df_filtered['altitude_grouped'] = df_filtered['altitude_drone'].apply(lambda x: round(x))

    # Plot boxplot with grouped altitudes
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_filtered['altitude_grouped'], y=df_filtered['distance_m'])
    plt.xlabel("Drone altitude (m) (grouped)")
    plt.ylabel("Error (m)")
    plt.title("Error of corrected waypoints as a function of altitude\n(Camera pitch = -89.9° or -90.0°)")
    plt.xticks(rotation=45)
    plt.grid()

    # Save the plot
    plot_file = 'error_vs_altitude_filtered.png'
    plt.savefig(plot_file)
    print(f"Plot saved as '{plot_file}'")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
