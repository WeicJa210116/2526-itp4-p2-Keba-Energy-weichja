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

def request_weather_data(start=datetime.datetime.today(), end=None,param="temperature_2m"):
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
    
    params = {
        "latitude": 48.3064,
        "longitude": 14.2861,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": param,
        "timezone": "Europe/Berlin"
    }
    
    response = cache_session.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.text}")
    
    #print(response.json())
    
    data = response.json()
    hourly_data = data.get("hourly", {})
    timestamps = hourly_data.get("time", [])
    
    results = []
    
    for timestamp in timestamps:
        results.append((hourly_data[param][timestamps.index(timestamp)], timestamp))
    
    
    print(results)
    
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
    
    plt.figure(figsize=(10, 6), layout="constrained")
    
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)

    start_time = min(times_dt)
    end_time = max(times_dt)
    
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}, End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
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
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}, End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
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

if __name__ == "__main__":
    
    START_DATE = "2026-04-20"
    END_DATE = "2026-04-28"
    LENGTH_DAYS = 1
    
    temparatur_data = request_weather_data(START_DATE, END_DATE, param="temperature_2m")
    plot_values_over_time(temparatur_data, title="Temperature Over Time", ylabel="Temperature in °C", filename="img/temperature_over_time.png", color="red")

    prices = request_awattar(START_DATE, END_DATE, length_days=LENGTH_DAYS)
    plot_values_over_time(prices, title="Wattar Prices Over Time", ylabel="Price in Cent/kWh", filename="img/awattar_prices_over_time.png")

    cloud_cover_data = request_weather_data(START_DATE, END_DATE, param="cloud_cover")
    plot_values_over_time(cloud_cover_data, title="Cloud Cover Over Time", ylabel="Cloud Cover in %", filename="img/cloud_cover_over_time.png", color="yellow")

    
    plot_multiple_values_combined(
        datasets=[
            {'values': prices, 'label': 'Price', 'color': 'green', 'ylabel': 'Price (Cent/kWh)'},
            #{'values': temparatur_data, 'label': 'Temperature', 'color': 'red', 'ylabel': 'Temperature (°C)'},
            {'values': cloud_cover_data, 'label': 'Cloud Cover', 'color': 'yellow', 'ylabel': 'Cloud Cover (%)'}
        ],
        title="Combined Plots",
        filename="img/combined_plots.png"
    )

    print("All tests passed!")
