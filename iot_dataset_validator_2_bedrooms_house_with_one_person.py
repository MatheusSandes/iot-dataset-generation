event_cache = {}
import pandas as pd
import networkx as nx
from datetime import datetime, time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from collections import defaultdict, deque

# === Load house architecture (as graph) ===
def load_house_graph():
    G = nx.Graph()
    # 2 bedrooms house connections based on provided graph
    G.add_edge("Bedroom1", "LivingRoom")
    G.add_edge("Bedroom2", "Bathroom")
    G.add_edge("Bathroom", "ServiceArea")
    G.add_edge("ServiceArea", "Kitchen")
    G.add_edge("LivingRoom", "Kitchen")

    sensor_locations = {
        # Bedroom1 sensors (primary bedroom where person sleeps)
        "motion_sensor_bedroom1": "Bedroom1",
        "temp_sensor_bedroom1": "Bedroom1",
        "smart_light_bedroom1": "Bedroom1",
        
        # Bedroom2 sensors (occasionally accessed)
        "motion_sensor_bedroom2": "Bedroom2",
        "temp_sensor_bedroom2": "Bedroom2",
        "smart_light_bedroom2": "Bedroom2",
        
        # LivingRoom sensors
        "motion_sensor_living": ["LivingRoom", "LivingRoom"],
        "temp_sensor_living": "LivingRoom",
        "smart_tv": "LivingRoom",
        "smart_light_living": "LivingRoom", 
        "smart_lock_front": "LivingRoom",
        
        # Kitchen sensors
        "temp_sensor_kitchen": "Kitchen",
        "smart_light_kitchen": "Kitchen",
        "smart_plug_fridge": "Kitchen",
        
        # ServiceArea sensors
        "motion_sensor_service": "ServiceArea",
        "temp_sensor_service": "ServiceArea"
        
        # Bathroom: no sensors
    }

    return G, sensor_locations

# === User behavior profile for one person in 2-bedroom house ===
user_profile = {
    "sleep_hours": (time(22, 30), time(6, 0)),  # Sleep: 22:30-06:00
    "night_motion_allowed": False,  # No motion during sleep
    "work_hours": (time(8, 0), time(17, 0)),  # Work: 08:00-17:00
    "work_days": [0, 1, 2, 3, 4],  # Monday to Friday (0=Monday, 6=Sunday)
    
    # Primary locations for activities
    "primary_bedroom": "Bedroom1",  # Person sleeps here
    "morning_locations": ["Bedroom1", "Kitchen"],  # Active 06:00-08:00
    "evening_locations": ["LivingRoom", "Kitchen"],  # Active 17:00-22:30
    "occasional_locations": ["Bedroom2", "ServiceArea"],  # Rarely used
    
    # Environment settings (Winter Brazil)
    "temp_range": (21.0, 26.0),
    "humidity_range": (40, 70),
    "temp_humidity_correlation": (-0.7, -0.9),  # Inverse correlation
    
    # Technical correlations
    "motion_temp_increase": (0.5, 1.5),  # Temperature increase after motion (°C)
    "motion_temp_delay": (15, 30),  # Delay in minutes for temp increase
    "motion_power_increase": (100, 300),  # Power increase after motion (W)
    "temp_noise": 0.1,  # ±0.1°C noise
    "power_noise": 0.01,  # ±1% noise
    "motion_false_positive": (0.001, 0.003)  # 0.1-0.3% false positive rate
}

# === Validation rules ===
def is_sleep_time(ts, sleep_range):
    """Check if timestamp is during sleep hours"""
    start, end = sleep_range
    return ts.time() >= start or ts.time() < end

def is_work_time(ts, work_hours, work_days):
    """Check if timestamp is during work hours on a work day"""
    # Check if it's a work day (0=Monday, 6=Sunday)
    if ts.weekday() not in work_days:
        return False
    
    # Check if it's during work hours
    start, end = work_hours
    current_time = ts.time()
    return start <= current_time <= end

def get_activity_period(ts):
    """Determine which activity period the timestamp falls into"""
    current_time = ts.time()
    
    if time(6, 0) <= current_time < time(8, 0):
        return "morning_active"
    elif time(8, 0) <= current_time < time(17, 0):
        return "work_time"
    elif time(17, 0) <= current_time <= time(22, 30):
        return "evening_active"
    elif current_time >= time(22, 30) or current_time < time(6, 0):
        return "sleep_time"
    else:
        return "unknown"

def validate_location_usage(location, activity_period):
    """Validate if location usage is appropriate for the activity period"""
    # During morning, should primarily use Bedroom1 and Kitchen
    if activity_period == "morning_active":
        return location in user_profile["morning_locations"] or location in user_profile["occasional_locations"]
    
    # During evening, should primarily use LivingRoom and Kitchen
    elif activity_period == "evening_active":
        return location in user_profile["evening_locations"] or location in user_profile["occasional_locations"]
    
    # During work time, no activity expected
    elif activity_period == "work_time":
        return False
    
    # During sleep time, minimal activity expected
    elif activity_period == "sleep_time":
        return False
    
    return True

def parse_timestamp(timestamp_str):
    """Robust timestamp parsing using pandas"""
    try:
        ts = pd.to_datetime(timestamp_str, errors='coerce', infer_datetime_format=True)
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None

def validate_row(row, graph, sensor_locations, previous_room, motion_tracker, activity_tracker):
    errors = []
    
    # Helper function to detect if there's motion in this row
    def has_motion(row):
        return (row["motion"] in [1, "true", "True"] or 
                row["event_type"] in ["motion_detected", "motion_trigger"])

    # Rule 1: Sensor must match room
    expected_location = sensor_locations.get(row["trigger_sensor"])
    if expected_location:
        # Handle both single location and list of acceptable locations
        if isinstance(expected_location, list):
            location_valid = row["location"] in expected_location
        else:
            location_valid = expected_location == row["location"]
        
        if not location_valid:
            errors.append(f"Sensor {row['trigger_sensor']} used in wrong room ({row['location']})")

    # Parse timestamp
    timestamp_str = row["timestamp"]
    ts = parse_timestamp(timestamp_str)
    
    if ts is None:
        errors.append(f"Invalid timestamp format: {timestamp_str}")
        return errors, previous_room
    
    # Get activity period for further validation
    activity_period = get_activity_period(ts)
    current_location = row["location"]
    
    # Rule 2: No events during sleep hours
    if is_sleep_time(ts, user_profile["sleep_hours"]):
        errors.append(f"Event during sleep hours: {timestamp_str} (sleep: {user_profile['sleep_hours'][0]}-{user_profile['sleep_hours'][1]})")
    
    # Rule 3: No events during work hours (house should be empty)
    if is_work_time(ts, user_profile["work_hours"], user_profile["work_days"]):
        errors.append(f"Event during work hours (house should be empty): {timestamp_str} (work: {user_profile['work_hours'][0]}-{user_profile['work_hours'][1]})")
    
    # Rule 4: Validate location usage patterns
    if not validate_location_usage(current_location, activity_period):
        if activity_period == "morning_active":
            errors.append(f"Unusual location for morning activity: {current_location} at {timestamp_str} (expected: {user_profile['morning_locations']})")
        elif activity_period == "evening_active":
            errors.append(f"Unusual location for evening activity: {current_location} at {timestamp_str} (expected: {user_profile['evening_locations']})")
    
    # Rule 5: Validate temperature and humidity ranges and correlation
    temp = row.get("temperature")
    humidity = row.get("humidity")
    if temp is not None and humidity is not None and temp != "" and humidity != "":
        try:
            temp = float(temp)
            humidity = float(humidity)
            temp_humidity_errors = validate_temp_humidity_correlation(temp, humidity)
            errors.extend(temp_humidity_errors)
        except (ValueError, TypeError):
            if temp != "" and temp is not None:
                errors.append(f"Invalid temperature value: {temp}")
            if humidity != "" and humidity is not None:
                errors.append(f"Invalid humidity value: {humidity}")
    
    # Track motion for frequency analysis
    if has_motion(row):
        motion_tracker['all_motions'].append(ts)
        
        # Store motion event details for correlation analysis
        motion_tracker['motion_events'].append({
            'timestamp': ts,
            'location': current_location,
            'event_id': row["event_id"]
        })
        
        # Track by time category
        if is_sleep_time(ts, user_profile["sleep_hours"]):
            motion_tracker['night_motions'].append(ts)
        elif is_work_time(ts, user_profile["work_hours"], user_profile["work_days"]):
            motion_tracker['work_motions'].append(ts)
        else:
            motion_tracker['day_motions'].append(ts)
        
        # Track by hour for visualization
        hour_key = ts.hour
        motion_tracker['hourly_motions'][hour_key] += 1
    
    # Track temperature events for correlation analysis
    if row["event_type"] == "temperature_reading" and temp is not None and temp != "":
        try:
            temp = float(temp)
            motion_tracker['temp_events'].append({
                'timestamp': ts,
                'location': current_location,
                'temperature': temp,
                'event_id': row["event_id"]
            })
        except (ValueError, TypeError):
            pass
    
    # Track power events for correlation analysis
    power = row.get("power_consumption")
    if power is not None and power != "":
        try:
            power = float(power)
            motion_tracker['power_events'].append({
                'timestamp': ts,
                'location': current_location,
                'power': power,
                'event_id': row["event_id"]
            })
        except (ValueError, TypeError):
            pass
    
    # Track all activities for frequency analysis
    activity_tracker['all_activities'].append(ts)
    
    # Rule 6: Movement between rooms must follow graph structure
    if previous_room and has_motion(row):
        if not graph.has_edge(previous_room, current_location) and previous_room != current_location:
            errors.append(f"Invalid transition: {previous_room} → {current_location} not connected in house layout")

    # Rule 7: Duplicate motion event in the same location at the same second
    ts_key = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else None
    if ts_key and row["event_type"] == "motion_detected":
        cache_key = f"{ts_key}_{row['location']}"
        if event_cache.get(cache_key):
            errors.append(f"Duplicate motion event at {row['location']} on {ts_key}")
        else:
            event_cache[cache_key] = True

    return errors, current_location if has_motion(row) else previous_room

def validate_temp_humidity_correlation(temp, humidity):
    """Validate temperature-humidity inverse correlation"""
    errors = []
    temp_range = user_profile["temp_range"]
    humidity_range = user_profile["humidity_range"]
    
    # Check if values are within expected ranges
    if not (temp_range[0] <= temp <= temp_range[1]):
        errors.append(f"Temperature {temp}°C outside expected range {temp_range[0]}-{temp_range[1]}°C")
    
    if not (humidity_range[0] <= humidity <= humidity_range[1]):
        errors.append(f"Humidity {humidity}% outside expected range {humidity_range[0]}-{humidity_range[1]}%")
    
    return errors

def validate_motion_temperature_response(motion_events, temp_events):
    """Validate that temperature increases after motion events within expected timeframe"""
    errors = []
    temp_increase_range = user_profile["motion_temp_increase"]
    delay_range = user_profile["motion_temp_delay"]
    
    motion_without_temp_response = 0
    
    for motion_event in motion_events:
        motion_time = motion_event['timestamp']
        motion_location = motion_event['location']
        
        # Look for temperature readings in the same location within delay window
        related_temp_events = []
        for temp_event in temp_events:
            if (temp_event['location'] == motion_location and 
                temp_event['timestamp'] > motion_time):
                time_diff = (temp_event['timestamp'] - motion_time).total_seconds() / 60  # minutes
                if delay_range[0] <= time_diff <= delay_range[1]:
                    related_temp_events.append(temp_event)
        
        if not related_temp_events:
            motion_without_temp_response += 1
    
    # Allow some tolerance - not every motion needs to have temp response
    if len(motion_events) > 0:
        missing_temp_percentage = (motion_without_temp_response / len(motion_events)) * 100
        if missing_temp_percentage > 70:  # More than 70% without temp response is suspicious
            errors.append(f"High percentage of motion events without temperature response: {missing_temp_percentage:.1f}% (expected < 70%)")
    
    return errors

def validate_motion_power_response(motion_events, power_events):
    """Validate that power consumption increases after motion events"""
    errors = []
    
    motion_without_power_response = 0
    
    for motion_event in motion_events:
        motion_time = motion_event['timestamp']
        motion_location = motion_event['location']
        
        # Look for power readings in the same location after motion
        related_power_events = []
        for power_event in power_events:
            if (power_event['location'] == motion_location and 
                power_event['timestamp'] > motion_time):
                time_diff = (power_event['timestamp'] - motion_time).total_seconds() / 60  # minutes
                if time_diff <= 5:  # Within 5 minutes
                    related_power_events.append(power_event)
        
        if not related_power_events:
            motion_without_power_response += 1
    
    # Allow some tolerance
    if len(motion_events) > 0:
        missing_power_percentage = (motion_without_power_response / len(motion_events)) * 100
        if missing_power_percentage > 60:  # More than 60% without power response is suspicious
            errors.append(f"High percentage of motion events without power response: {missing_power_percentage:.1f}% (expected < 60%)")
    
    return errors

# === Frequency analysis functions ===
def analyze_motion_frequency(motion_tracker):
    """Analyze motion patterns and detect anomalies based on frequency"""
    errors = []
    
    total_motions = len(motion_tracker['all_motions'])
    night_motions = len(motion_tracker['night_motions'])
    work_motions = len(motion_tracker['work_motions'])
    day_motions = len(motion_tracker['day_motions'])
    
    if total_motions == 0:
        return errors
    
    # Calculate frequency percentages
    night_percentage = (night_motions / total_motions) * 100
    work_percentage = (work_motions / total_motions) * 100
    
    # Rule 8: Night motion frequency analysis
    # For one person, there should be no motion during sleep hours
    if night_motions > 0:
        errors.append(f"Motion detected during sleep hours: {night_motions} events ({night_percentage:.1f}% of total)")
    
    # Rule 9: Work hours motion analysis
    # For one person, there should be no motion during work hours
    if work_motions > 0:
        errors.append(f"Motion detected during work hours: {work_motions} events ({work_percentage:.1f}% of total)")
    
    return errors

def create_motion_frequency_graph(motion_tracker, output_file="motion_frequency_analysis_2_bedrooms_house_with_one_person.png"):
    """Create a line graph showing motion frequency throughout the day"""
    hourly_motions = motion_tracker['hourly_motions']
    
    # Create arrays for plotting
    hours = list(range(24))
    motion_counts = [hourly_motions[h] for h in hours]
    
    plt.figure(figsize=(12, 6))
    plt.plot(hours, motion_counts, 'b-', linewidth=2, label='Motion Events')
    plt.fill_between(hours, motion_counts, alpha=0.3)
    
    # Add sleep time shading
    sleep_start, sleep_end = user_profile["sleep_hours"]
    sleep_start_hour = sleep_start.hour + sleep_start.minute/60
    sleep_end_hour = sleep_end.hour + sleep_end.minute/60
    
    # Handle sleep time that crosses midnight
    if sleep_start_hour > sleep_end_hour:
        plt.axvspan(sleep_start_hour, 24, alpha=0.2, color='red', label='Sleep Time')
        plt.axvspan(0, sleep_end_hour, alpha=0.2, color='red')
    else:
        plt.axvspan(sleep_start_hour, sleep_end_hour, alpha=0.2, color='red', label='Sleep Time')
    
    # Add work time shading
    work_start, work_end = user_profile["work_hours"]
    work_start_hour = work_start.hour + work_start.minute/60
    work_end_hour = work_end.hour + work_end.minute/60
    plt.axvspan(work_start_hour, work_end_hour, alpha=0.2, color='orange', label='Work Time')
    
    # Add activity period markings
    plt.axvspan(6, 8, alpha=0.1, color='green', label='Morning Active')
    plt.axvspan(17, 22.5, alpha=0.1, color='blue', label='Evening Active')
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Motion Events')
    plt.title('Motion Frequency Throughout the Day - 2 Bedrooms House (One Person)')
    plt.xticks(hours)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def get_motion_statistics(motion_tracker):
    """Get detailed statistics about motion patterns"""
    total_motions = len(motion_tracker['all_motions'])
    night_motions = len(motion_tracker['night_motions'])
    work_motions = len(motion_tracker['work_motions'])
    day_motions = len(motion_tracker['day_motions'])
    
    stats = {
        'total_motions': total_motions,
        'night_motions': night_motions,
        'work_motions': work_motions,
        'day_motions': day_motions,
        'night_percentage': (night_motions / total_motions) * 100 if total_motions > 0 else 0,
        'work_percentage': (work_motions / total_motions) * 100 if total_motions > 0 else 0,
        'day_percentage': (day_motions / total_motions) * 100 if total_motions > 0 else 0
    }
    
    return stats

# === Main validation loop ===
def validate_csv(filepath, generate_graph=True):
    global event_cache
    event_cache.clear()  # Clear cache for each validation run
    
    df = pd.read_csv(filepath)
    graph, sensor_locations = load_house_graph()
    error_log = []
    error_count = 0
    previous_room = None

    # Initialize motion and activity trackers
    motion_tracker = {
        'all_motions': [],
        'night_motions': [],
        'work_motions': [],
        'day_motions': [],
        'hourly_motions': defaultdict(int),
        'motion_events': [],
        'temp_events': [],
        'power_events': []
    }
    
    activity_tracker = {
        'all_activities': []
    }
    
    seen_event_ids = set()  # Track unique event IDs

    # Process each row
    for _, row in df.iterrows():
        row_errors, previous_room = validate_row(row, graph, sensor_locations, previous_room, motion_tracker, activity_tracker)
        
        # Rule 10: Check for duplicate event IDs
        event_id = row["event_id"]
        if event_id in seen_event_ids:
            row_errors.append(f"Duplicate event ID: {event_id}")
        else:
            seen_event_ids.add(event_id)
        
        if row_errors:
            error_count += len(row_errors)
            error_log.append({
                "timestamp": row["timestamp"],
                "location": row["location"],
                "event_id": event_id,
                "errors": row_errors
            })

    # Perform frequency analysis after processing all data
    frequency_errors = analyze_motion_frequency(motion_tracker)
    
    # Add motion-temperature correlation analysis
    motion_temp_errors = validate_motion_temperature_response(
        motion_tracker['motion_events'], 
        motion_tracker['temp_events']
    )
    
    # Add motion-power correlation analysis
    motion_power_errors = validate_motion_power_response(
        motion_tracker['motion_events'], 
        motion_tracker['power_events']
    )
    
    # Combine all analysis errors
    analysis_errors = frequency_errors + motion_temp_errors + motion_power_errors
    
    # Add analysis errors to the log
    for analysis_error in analysis_errors:
        error_count += 1
        error_log.append({
            "timestamp": "ANALYSIS",
            "location": "OVERALL",
            "event_id": "ANALYSIS_" + str(len(error_log)),
            "errors": [analysis_error]
        })
    
    # Generate motion frequency graph if requested
    graph_file = None
    if generate_graph:
        try:
            graph_file = create_motion_frequency_graph(motion_tracker)
        except Exception as e:
            print(f"Warning: Could not generate motion frequency graph: {e}")
    
    # Get motion statistics
    motion_stats = get_motion_statistics(motion_tracker)

    return error_count, error_log, motion_stats, graph_file

# === Example usage ===
if __name__ == "__main__":
    csv_path = "iot_dataset_2_bedrooms_house_with_one_person_errors.csv"  # Change this path to your CSV file
    error_count, error_log, motion_stats, graph_file = validate_csv(csv_path)

    print(f"\n=== IoT Dataset Validation Results ===")
    print(f"Profile: One Person in 2 Bedrooms House")
    print(f"Primary bedroom: Bedroom1 (where person sleeps)")
    print(f"Morning locations: Bedroom1, Kitchen (06:00-08:00)")
    print(f"Evening locations: LivingRoom, Kitchen (17:00-22:30)")
    print(f"Occasional locations: Bedroom2, ServiceArea")
    print(f"Sleep hours: 22:30-06:00 (no events allowed)")
    print(f"Work hours: 08:00-17:00 (no events allowed - house empty)")
    print(f"Environment: Winter Brazil, 21-26°C, 40-70% humidity")
    print(f"Correlations: Motion→Temp (+0.5-1.5°C), Motion→Power (+100-300W)")
    
    print(f"\n=== Motion Statistics ===")
    print(f"Total motions: {motion_stats['total_motions']}")
    print(f"Day motions: {motion_stats['day_motions']} ({motion_stats['day_percentage']:.1f}%)")
    print(f"Night motions: {motion_stats['night_motions']} ({motion_stats['night_percentage']:.1f}%)")
    print(f"Work motions: {motion_stats['work_motions']} ({motion_stats['work_percentage']:.1f}%)")
    
    if graph_file:
        print(f"\n=== Motion Frequency Graph ===")
        print(f"Graph saved to: {graph_file}")
    
    print(f"\n=== Total Errors Found: {error_count} ===")
    print("\n=== Detailed Errors ===")
    for entry in error_log:
        print(f"{entry['timestamp']} - {entry['location']} -> {entry['errors']}")
        
    # Print summary by error type
    print(f"\n=== Error Summary ===")
    error_types = {}
    for entry in error_log:
        for error in entry['errors']:
            error_type = error.split(':')[0] if ':' in error else error.split(' ')[0]
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    for error_type, count in sorted(error_types.items()):
        print(f"{error_type}: {count} occurrences") 