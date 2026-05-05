import requests

import numpy as np
import scipy.stats as scs
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import datetime
import requests_cache
import openmeteo_requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter 

def request_weather_data(start=datetime.datetime.today(), end=None,hourly_params="temperature_2m", lat=48.3064, lon=14.2861):
    """
    Request weather data from the Open-Meteo API for a specific date range.
    Parameters:
    - start: The optional starting date for the data request (default is today's date). Can be a string in 'YYYY-MM-DD' format or a datetime object.
    - end: The optional end date for the data request. Can be a string in 'YYYY-MM-DD' format or a datetime object. If not provided, the API will return data starting from the specified start date until the latest available data.
    Returns:
    - A list of tuples containing the temperature and corresponding timestamp for each entry in the response
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry = Retry(total=5, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    cache_session.mount('http://', adapter)
    cache_session.mount('https://', adapter)
    
    start_date = start.strftime('%Y-%m-%d') if isinstance(start, datetime.datetime) else start
    end_date = end.strftime('%Y-%m-%d') if isinstance(end, datetime.datetime) else end
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    paramStr = hourly_params if isinstance(hourly_params, str) else ",".join(hourly_params)
          
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": paramStr,
        "timezone": "Europe/Berlin"
    }
    
    response = cache_session.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.text}")
    
    #print(response.json())
    
    data = response.json()
    hourly_data = data.get("hourly", {})
    timestamps = hourly_data.get("time", [])
    
    results = {}
    for param in hourly_params:
        if param not in hourly_data:
            print(f"Warning: '{param}' not found in response data")
        results[param] = [] 
    
    # Convert hourly_params to list if it's a string
    params_list = hourly_params if isinstance(hourly_params, list) else [hourly_params]
    
    for param in params_list:
        if param in hourly_data:
            param_values = hourly_data[param]
            for idx, timestamp in enumerate(timestamps):
                if idx < len(param_values):
                    value = param_values[idx]
                    if value is not None:
                        results[param].append((value, timestamp))
    
    
    #print(results)
    
    return results


def request_awattar(start=datetime.datetime.today(),end=None,length_days=1):
    """
    Request price data from the aWATTar API for a specific day and optional end date.
    Parameters:
    - day: The optional starting date for the data request (default is today's date). Can be a string in 'YYYY-MM-DD' format or a datetime object.
    - end: The optional end date for the data request. Can be a string in 'YYYY-MM-DD' format or a datetime object. If not provided, the API will return data starting from the specified day until the latest available data.
    - length_days: The number of days for which to request price data (default is 1).
    Returns:
    - A list of tuples containing the market price and corresponding timestamp for each entry in the response
    """
    if isinstance(start, str):
        start = pd.to_datetime(start)
    
    #print(f"Requesting data for start: {start.strftime('%Y-%m-%d')}")
    start = pd.to_datetime(start).timestamp() * 1000
    
    if end is not None:
        if isinstance(end, str):
            end = pd.to_datetime(end)
        #print(f"Requesting data for end: {end.strftime('%Y-%m-%d')}")
        end = pd.to_datetime(end).timestamp() * 1000
        end += 24 * 60 * 60 * 1000 * length_days  # Add the specified number of days to the end timestamp
        url = f"https://api.awattar.de/v1/marketdata?start={start}&end={end}"
    else:
        url = f"https://api.awattar.de/v1/marketdata?start={start}"
    response = requests.get(url)
    
    assert response.status_code == 200, "API request failed with status code: {}".format(response.status_code)
    
    data = response.json()
    
    assert "data" in data, "Response JSON does not contain 'data' key"
    assert isinstance(data["data"], list), "'data' key is not a list"
    
    
    prices = []
    for entry in data["data"]:
        #print(f"Time: {pd.to_datetime(entry['start_timestamp'], unit='ms').strftime('%Y-%m-%d %H:%M:%S')}, Price: {entry['marketprice']}")

        time = entry["start_timestamp"]
        price = entry["marketprice"]
        
        
        prices.append((price, time))
    
    return prices


def plot_values_over_time(values, title="Values Over Time", ylabel="Value", filename="img/values_over_time.png",color="blue"):
    
    
    for p in values:
        if not isinstance(p, tuple) or len(p) != 2:
            print(f"Error: Invalid data format for entry {p}. Expected a tuple of (value, timestamp).")
            return
    
    if not values:
        print("Error: No data to plot")
        return
    
    times = [p[1] for p in values]
    values = [p[0] for p in values]
    
    
    # Convert timestamps to datetime objects - handle both milliseconds and ISO format
    times_dt = []
    for t in times:
        if isinstance(t, str):
            times_dt.append(pd.to_datetime(t))
        else:
            times_dt.append(pd.to_datetime(t, unit='ms'))
    
    if not times_dt:
        print("Error: No valid timestamps to plot")
        return
    
    # Filter out None values
    times_values_pairs = [(t, v) for t, v in zip(times_dt, values) if v is not None]
    if not times_values_pairs:
        print("Error: No valid values to plot")
        return
    
    times_dt = [t for t, v in times_values_pairs]
    values = [v for t, v in times_values_pairs]
    
    plt.figure(figsize=(10, 6), layout="constrained")
    
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)

    start_time = min(times_dt)
    end_time = max(times_dt)
    
    #print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}, End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    plt.xlim(start_time, end_time)
    
    plt.ylim(min(values) * 0.9, max(values) * 1.1)
    
    plt.plot(times_dt, values, marker='o', linestyle='-', alpha=1, color=color, linewidth=2)
    
    # Format x-axis timestamps
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.xticks(rotation=45, ha='right')
     
    plt.savefig(filename)
    print(f"Plot saved to {filename}")


def plot_multiple_values_combined(datasets, title="Combined Plot", filename="img/combined_plot.png"):
    """
    Plot multiple datasets in a single plot with multiple y-axes.
    
    Parameters:
    - datasets: List of dictionaries with keys: 'values', 'label', 'color', 'ylabel'
      Example: [
        {'values': prices, 'label': 'Price', 'color': 'green', 'ylabel': 'Price (Cent/kWh)'},
        {'values': temps, 'label': 'Temperature', 'color': 'red', 'ylabel': 'Temperature (°C)'}
      ]
    - title: Title of the plot
    - filename: Output filename
    """
    if not datasets or not any(datasets):
        print("Error: No data to plot")
        return
    
    fig, ax1 = plt.subplots(figsize=(12, 6), layout="constrained")
    
    plt.title(title)
    
    # Track all times for x-axis limits
    all_times_dt = []
    
    # Plot first dataset on primary y-axis
    first_dataset = datasets[0]
    times = [p[1] for p in first_dataset['values']]
    values = [p[0] for p in first_dataset['values']]
    
    times_dt = []
    for t in times:
        if isinstance(t, str):
            times_dt.append(pd.to_datetime(t))
        else:
            times_dt.append(pd.to_datetime(t, unit='ms'))
    
    all_times_dt.extend(times_dt)
    
    color = first_dataset.get('color', 'blue')
    ax1.plot(times_dt, values, marker='o', linestyle='-', alpha=0.8, color=color, linewidth=2, label=first_dataset['label'])
    ax1.set_xlabel("Time")
    ax1.set_ylabel(first_dataset.get('ylabel', 'Values'), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Plot additional datasets on separate y-axes
    axes = [ax1]
    for i, dataset in enumerate(datasets[1:], 1):
        if not dataset.get('values'):
            continue
            
        ax = ax1.twinx()
        ax.spines['right'].set_position(('outward', 60 * i))
        
        times = [p[1] for p in dataset['values']]
        values = [p[0] for p in dataset['values']]
        
        times_dt = []
        for t in times:
            if isinstance(t, str):
                times_dt.append(pd.to_datetime(t))
            else:
                times_dt.append(pd.to_datetime(t, unit='ms'))
        
        all_times_dt.extend(times_dt)
        
        color = dataset.get('color', f'C{i}')
        ax.plot(times_dt, values, marker='s', linestyle='--', alpha=0.8, color=color, linewidth=2, label=dataset['label'])
        ax.set_ylabel(dataset.get('ylabel', f'Values {i}'), color=color)
        ax.tick_params(axis='y', labelcolor=color)
        axes.append(ax)
    
    # Set x-axis limits
    if all_times_dt:
        start_time = min(all_times_dt)
        end_time = max(all_times_dt)
        ax1.set_xlim(start_time, end_time)
        #print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}, End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Format x-axis timestamps
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    ax1.tick_params(axis='x', rotation=45)
    
    # Add legend
    lines = []
    labels = []
    for ax in axes:
        line, label = ax.get_legend_handles_labels()
        lines.extend(line)
        labels.extend(label)
    
    if lines:
        fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(datasets))
    
    plt.savefig(filename, bbox_inches='tight')
    print(f"Combined plot saved to {filename}")


def save_to_csv(data, filename="data.csv"):
    df = pd.DataFrame(data, columns=["Value", "Timestamp"])
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def read_from_csv(filename="data.csv"):
    df = pd.read_csv(filename)
    return list(zip(df["Value"], df["Timestamp"]))

if __name__ == "__main__":
    
    # https://open-meteo.com/en/docs?bounding_box=-90,-180,90,180&hourly=temperature_2m,weather_code,relative_humidity_2m,apparent_temperature,rain,showers,precipitation_probability,snowfall,cloud_cover,cloud_cover_mid,cloud_cover_low,cloud_cover_high,visibility,wind_speed_10m,wind_speed_80m,wind_speed_120m,wind_speed_180m,wind_direction_10m,wind_direction_80m,wind_direction_120m,wind_direction_180m,temperature_80m,temperature_120m,temperature_180m
    
    START_DATE = "2026-04-28"
    END_DATE = "2026-04-29"
    LENGTH_DAYS = 1
    PUPU_PARK_LOCATION = { "latitude": 48.31189543738503, "longitude": 14.244101450717908 }
    WEATHER_PARAMS = ["temperature_2m", "cloudcover", "windspeed_10m", "temperature_2m", "weather_code", "relative_humidity_2m", "apparent_temperature", "rain", "showers", "precipitation_probability", "snowfall", "cloud_cover", "cloud_cover_mid", "cloud_cover_low", "cloud_cover_high", "visibility", "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m", "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m", "temperature_80m", "temperature_120m", "temperature_180m"]
    

    prices = request_awattar(START_DATE, END_DATE, length_days=LENGTH_DAYS)

    weather_data = request_weather_data(START_DATE, END_DATE, hourly_params=WEATHER_PARAMS, lat=PUPU_PARK_LOCATION["latitude"], lon=PUPU_PARK_LOCATION["longitude"])

    save_to_csv(prices, filename="data/awattar_price_data.csv")
    plot_values_over_time(prices, title="aWATTar Price Over Time", ylabel="Price (Cent/kWh)", filename="img/awattar_price_over_time.png", color="green")

    
    for param in WEATHER_PARAMS:
        if param in weather_data:
            plot_values_over_time(weather_data[param], title=f"{param.capitalize()} Over Time", ylabel=param.capitalize(), filename=f"img/{param}_over_time.png", color="orange")
    for param in WEATHER_PARAMS:
        if param in weather_data:
            save_to_csv(weather_data[param], filename=f"data/{param}_data.csv")
   
    datasets = []
    colorlist = {
        "temperature_2m": "red",
        "cloudcover": "gray",
        "windspeed_10m": "blue",
        "weather_code": "purple",
        "relative_humidity_2m": "cyan",
        "apparent_temperature": "magenta",
        "rain": "navy",
        "showers": "teal",
        "precipitation_probability": "olive",
        "snowfall": "brown",
        "cloud_cover": "lightgray",
        "cloud_cover_mid": "darkgray",
        "cloud_cover_low": "dimgray",
        "cloud_cover_high": "gainsboro",
        "visibility": "goldenrod",
        "wind_speed_10m": "cornflowerblue",
        "wind_speed_80m": "steelblue",
        "wind_speed_120m": "royalblue",
        "wind_speed_180m": "dodgerblue",
        "wind_direction_10m": "sandybrown",
        "wind_direction_80m": "peru",
        "wind_direction_120m": "chocolate",
        "wind_direction_180m": "sienna",
        "temperature_80m": "salmon",
        "temperature_120m": "lightcoral",
        "temperature_180m": "indianred"
    }
    for param in WEATHER_PARAMS:
        if param in weather_data:
            color = colorlist[param] if param in colorlist else "orange"
            datasets.append({'values': weather_data[param], 'label': param.capitalize(), 'color': color, 'ylabel': param.capitalize()})

    plot_multiple_values_combined(
           datasets=datasets,
       title="Combined Plots",
       filename="img/combined_plots.png"
    )

    print("All tests passed!")
