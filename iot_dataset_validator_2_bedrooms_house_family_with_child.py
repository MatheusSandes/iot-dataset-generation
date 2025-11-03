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
    # 2 bedrooms house connections for family with child
    G.add_edge("Bedroom1", "LivingRoom")
    G.add_edge("Bedroom2", "Bathroom")
    G.add_edge("Bathroom", "ServiceArea")
    G.add_edge("ServiceArea", "Kitchen")
    G.add_edge("LivingRoom", "Kitchen")

    sensor_locations = {
        # Bedroom1 sensors (parents' bedroom)
        "motion_sensor_bedroom1": "Bedroom1",
        "temp_sensor_bedroom1": "Bedroom1",
        "smart_light_bedroom1": "Bedroom1",
        
        # Bedroom2 sensors (child's bedroom)
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

# === User behavior profile for family with child ===
user_profile = {
    # Adult 1: wakes 06:00, leaves 08:00, returns 17:00, sleeps 22:30
    # Adult 2: wakes 07:00, leaves 09:00, returns 18:00, sleeps 23:00
    # Child: wakes 06:30, leaves for school 07:30, returns 17:30, sleeps 22:00
    
    "all_sleep_hours": (time(23, 0), time(6, 0)),  # All asleep: 23:00-06:00
    "house_empty_hours": (time(9, 0), time(17, 0)),  # House empty: 09:00-17:00
    "work_days": [0, 1, 2, 3, 4],  # Monday to Friday (0=Monday, 6=Sunday)
    "night_motion_allowed": True,  # Child may wake up during night
    "night_motion_threshold": 25,  # Up to 25% night motion acceptable for family
    
    # Activity periods based on family schedule
    "adult1_solo_morning": (time(6, 0), time(6, 30)),       # 06:00-06:30: Only Adult 1
    "adult1_child_morning": (time(6, 30), time(7, 0)),      # 06:30-07:00: Adult 1 + Child
    "all_active_morning": (time(7, 0), time(7, 30)),        # 07:00-07:30: All three active
    "adults_active_preparation": (time(7, 30), time(8, 0)), # 07:30-08:00: Both adults (child at school)
    "adult2_solo_departure": (time(8, 0), time(9, 0)),      # 08:00-09:00: Only Adult 2
    "adult1_solo_return": (time(17, 0), time(17, 30)),      # 17:00-17:30: Only Adult 1
    "adult1_child_afternoon": (time(17, 30), time(18, 0)),  # 17:30-18:00: Adult 1 + Child
    "all_active_evening": (time(18, 0), time(22, 0)),       # 18:00-22:00: All three active
    "adults_only_evening": (time(22, 0), time(22, 30)),     # 22:00-22:30: Both adults (child sleeping)
    "adult2_solo_night": (time(22, 30), time(23, 0)),       # 22:30-23:00: Only Adult 2
    
    # Living arrangements
    "parents_bedroom": "Bedroom1",  # Adults sleep here
    "child_bedroom": "Bedroom2",    # Child sleeps here
    "family_areas": ["LivingRoom", "Kitchen"],  # Main family activity areas
    "occasional_areas": ["ServiceArea"],  # Less frequent use
    
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
def is_all_sleep_time(ts, sleep_range):
    """Check if timestamp is during hours when all family members are asleep"""
    start, end = sleep_range
    return ts.time() >= start or ts.time() < end

def is_house_empty_time(ts, empty_hours, work_days):
    """Check if house should be completely empty (all at work/school)"""
    # Check if it's a work day (0=Monday, 6=Sunday)
    if ts.weekday() not in work_days:
        return False
    
    # Check if it's during house empty hours
    start, end = empty_hours
    current_time = ts.time()
    return start <= current_time <= end

def get_activity_period(ts):
    """Determine which activity period the timestamp falls into"""
    current_time = ts.time()
    
    # Check each period based on family schedule
    if user_profile["adult1_solo_morning"][0] <= current_time < user_profile["adult1_solo_morning"][1]:
        return "adult1_solo_morning"
    elif user_profile["adult1_child_morning"][0] <= current_time < user_profile["adult1_child_morning"][1]:
        return "adult1_child_morning"
    elif user_profile["all_active_morning"][0] <= current_time < user_profile["all_active_morning"][1]:
        return "all_active_morning"
    elif user_profile["adults_active_preparation"][0] <= current_time < user_profile["adults_active_preparation"][1]:
        return "adults_active_preparation"
    elif user_profile["adult2_solo_departure"][0] <= current_time < user_profile["adult2_solo_departure"][1]:
        return "adult2_solo_departure"
    elif user_profile["adult1_solo_return"][0] <= current_time < user_profile["adult1_solo_return"][1]:
        return "adult1_solo_return"
    elif user_profile["adult1_child_afternoon"][0] <= current_time < user_profile["adult1_child_afternoon"][1]:
        return "adult1_child_afternoon"
    elif user_profile["all_active_evening"][0] <= current_time <= user_profile["all_active_evening"][1]:
        return "all_active_evening"
    elif user_profile["adults_only_evening"][0] <= current_time < user_profile["adults_only_evening"][1]:
        return "adults_only_evening"
    elif user_profile["adult2_solo_night"][0] <= current_time < user_profile["adult2_solo_night"][1]:
        return "adult2_solo_night"
    elif is_house_empty_time(ts, user_profile["house_empty_hours"], user_profile["work_days"]):
        return "house_empty"
    elif is_all_sleep_time(ts, user_profile["all_sleep_hours"]):
        return "all_sleep"
    else:
        return "unknown"

def validate_activity_pattern(location, activity_period):
    """Validate if location usage is appropriate for the activity period"""
    # During periods with people present, all areas can be used
    if activity_period in ["all_active_morning", "all_active_evening"]:
        return True  # All family members active - all areas valid
    
    # During periods with some family members present
    elif activity_period in ["adult1_solo_morning", "adult1_child_morning", "adults_active_preparation", 
                           "adult2_solo_departure", "adult1_solo_return", "adult1_child_afternoon", 
                           "adults_only_evening", "adult2_solo_night"]:
        return True  # Some family members active - all areas still valid
    
    # During house empty time, no activity expected
    elif activity_period == "house_empty":
        return False
    
    # During sleep time, activity allowed in bedrooms (child may wake up)
    elif activity_period == "all_sleep":
        return location in [user_profile["parents_bedroom"], user_profile["child_bedroom"]]
    
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
    
    # Rule 2: No events during house empty hours
    if is_house_empty_time(ts, user_profile["house_empty_hours"], user_profile["work_days"]):
        errors.append(f"Event during house empty hours: {timestamp_str} (empty: {user_profile['house_empty_hours'][0]}-{user_profile['house_empty_hours'][1]})")
    
    # Rule 3: Limited events during all sleep hours
    if is_all_sleep_time(ts, user_profile["all_sleep_hours"]):
        # For family with child, night motion is allowed but should be in bedrooms
        if has_motion(row) and current_location not in [user_profile["parents_bedroom"], user_profile["child_bedroom"]]:
            errors.append(f"Motion outside bedrooms during sleep hours: {current_location} at {timestamp_str} (expected only in bedrooms)")
        elif not has_motion(row) and row["event_type"] not in ["temperature_reading", "humidity_reading"]:
            # Non-motion events during sleep should mostly be environmental or in bedrooms
            if current_location not in [user_profile["parents_bedroom"], user_profile["child_bedroom"]]:
                errors.append(f"Non-environmental event outside bedrooms during sleep: {current_location} at {timestamp_str}")
    
    # Rule 4: Validate activity patterns
    if not validate_activity_pattern(current_location, activity_period):
        if activity_period == "house_empty":
            errors.append(f"Activity during house empty period: {current_location} at {timestamp_str}")
        elif activity_period == "all_sleep":
            errors.append(f"Activity outside bedrooms during sleep: {current_location} at {timestamp_str}")
    
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
        if is_all_sleep_time(ts, user_profile["all_sleep_hours"]):
            motion_tracker['night_motions'].append(ts)
        elif is_house_empty_time(ts, user_profile["house_empty_hours"], user_profile["work_days"]):
            motion_tracker['empty_house_motions'].append(ts)
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
        if missing_temp_percentage > 75:  # More than 75% without temp response is suspicious (more tolerant for family)
            errors.append(f"High percentage of motion events without temperature response: {missing_temp_percentage:.1f}% (expected < 75%)")
    
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
    
    # Allow some tolerance - for family with child, power correlation is more variable
    if len(motion_events) > 0:
        missing_power_percentage = (motion_without_power_response / len(motion_events)) * 100
        if missing_power_percentage > 65:  # More than 65% without power response is suspicious (more tolerant for family)
            errors.append(f"High percentage of motion events without power response: {missing_power_percentage:.1f}% (expected < 65%)")
    
    return errors

# === Frequency analysis functions ===
def analyze_motion_frequency(motion_tracker):
    """Analyze motion patterns and detect anomalies based on frequency"""
    errors = []
    
    total_motions = len(motion_tracker['all_motions'])
    night_motions = len(motion_tracker['night_motions'])
    empty_house_motions = len(motion_tracker['empty_house_motions'])
    day_motions = len(motion_tracker['day_motions'])
    
    if total_motions == 0:
        return errors
    
    # Calculate frequency percentages
    night_percentage = (night_motions / total_motions) * 100
    empty_house_percentage = (empty_house_motions / total_motions) * 100
    
    # Rule 8: Night motion frequency analysis
    # For family with child, night motion is more acceptable (child may wake up)
    night_threshold = user_profile["night_motion_threshold"]
    if night_percentage > night_threshold:
        errors.append(f"High night motion frequency: {night_percentage:.1f}% of total motion during sleep hours (expected < {night_threshold}%)")
    
    # Rule 9: Empty house motion analysis
    # Should be no motion when house is empty
    if empty_house_motions > 0:
        errors.append(f"Motion detected during empty house hours: {empty_house_motions} events ({empty_house_percentage:.1f}% of total)")
    
    return errors

def create_motion_frequency_graph(motion_tracker, output_file="motion_frequency_analysis_2_bedrooms_house_family_with_child.png"):
    """Create a line graph showing motion frequency throughout the day"""
    hourly_motions = motion_tracker['hourly_motions']
    
    # Create arrays for plotting
    hours = list(range(24))
    motion_counts = [hourly_motions[h] for h in hours]
    
    plt.figure(figsize=(14, 8))
    plt.plot(hours, motion_counts, 'b-', linewidth=2, label='Motion Events')
    plt.fill_between(hours, motion_counts, alpha=0.3)
    
    # Add sleep time shading
    sleep_start, sleep_end = user_profile["all_sleep_hours"]
    sleep_start_hour = sleep_start.hour + sleep_start.minute/60
    sleep_end_hour = sleep_end.hour + sleep_end.minute/60
    
    # Handle sleep time that crosses midnight
    if sleep_start_hour > sleep_end_hour:
        plt.axvspan(sleep_start_hour, 24, alpha=0.2, color='red', label='All Sleep')
        plt.axvspan(0, sleep_end_hour, alpha=0.2, color='red')
    else:
        plt.axvspan(sleep_start_hour, sleep_end_hour, alpha=0.2, color='red', label='All Sleep')
    
    # Add house empty time shading
    empty_start, empty_end = user_profile["house_empty_hours"]
    empty_start_hour = empty_start.hour + empty_start.minute/60
    empty_end_hour = empty_end.hour + empty_end.minute/60
    plt.axvspan(empty_start_hour, empty_end_hour, alpha=0.2, color='orange', label='House Empty')
    
    # Add family activity period markings
    plt.axvspan(6, 6.5, alpha=0.1, color='green', label='A1 Solo Morning')
    plt.axvspan(6.5, 7, alpha=0.1, color='cyan', label='A1+Child Morning')
    plt.axvspan(7, 7.5, alpha=0.1, color='blue', label='All Active Morning')
    plt.axvspan(7.5, 8, alpha=0.1, color='purple', label='Adults Preparation')
    plt.axvspan(8, 9, alpha=0.1, color='magenta', label='A2 Solo Departure')
    plt.axvspan(17, 17.5, alpha=0.1, color='lime', label='A1 Solo Return')
    plt.axvspan(17.5, 18, alpha=0.1, color='yellow', label='A1+Child Afternoon')
    plt.axvspan(18, 22, alpha=0.1, color='blue', label='All Active Evening')
    plt.axvspan(22, 22.5, alpha=0.1, color='pink', label='Adults Only Evening')
    plt.axvspan(22.5, 23, alpha=0.1, color='gray', label='A2 Solo Night')
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Motion Events')
    plt.title('Motion Frequency Throughout the Day - 2 Bedrooms House (Family with Child)')
    plt.xticks(hours)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def get_motion_statistics(motion_tracker):
    """Get detailed statistics about motion patterns"""
    total_motions = len(motion_tracker['all_motions'])
    night_motions = len(motion_tracker['night_motions'])
    empty_house_motions = len(motion_tracker['empty_house_motions'])
    day_motions = len(motion_tracker['day_motions'])
    
    stats = {
        'total_motions': total_motions,
        'night_motions': night_motions,
        'empty_house_motions': empty_house_motions,
        'day_motions': day_motions,
        'night_percentage': (night_motions / total_motions) * 100 if total_motions > 0 else 0,
        'empty_house_percentage': (empty_house_motions / total_motions) * 100 if total_motions > 0 else 0,
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
        'empty_house_motions': [],
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
    csv_path = "iot_dataset_2_bedrooms_house_family_with_child_errors.csv"  # Change this path to your CSV file
    error_count, error_log, motion_stats, graph_file = validate_csv(csv_path)

    print(f"\n=== IoT Dataset Validation Results ===")
    print(f"Profile: Family with Child in 2 Bedrooms House")
    print(f"Adult 1: Wakes 06:00, leaves 08:00, returns 17:00, sleeps 22:30")
    print(f"Adult 2: Wakes 07:00, leaves 09:00, returns 18:00, sleeps 23:00")
    print(f"Child: Wakes 06:30, school 07:30, returns 17:30, sleeps 22:00")
    print(f"Adults sleep in Bedroom1, Child sleeps in Bedroom2")
    print(f"")
    print(f"Activity Timeline:")
    print(f"  06:00-06:30 → Only Adult 1 active")
    print(f"  06:30-07:00 → Adult 1 + Child active")
    print(f"  07:00-07:30 → All three active")
    print(f"  07:30-08:00 → Both adults active (child at school)")
    print(f"  08:00-09:00 → Only Adult 2 active")
    print(f"  09:00-17:00 → House empty (all at work/school)")
    print(f"  17:00-17:30 → Only Adult 1 active")
    print(f"  17:30-18:00 → Adult 1 + Child active")
    print(f"  18:00-22:00 → All three active")
    print(f"  22:00-22:30 → Both adults active (child sleeping)")
    print(f"  22:30-23:00 → Only Adult 2 active")
    print(f"  23:00-06:00 → All asleep")
    print(f"")
    print(f"Environment: Winter Brazil, 21-26°C, 40-70% humidity")
    print(f"Correlations: Motion→Temp (+0.5-1.5°C), Motion→Power (+100-300W)")
    print(f"Night motion threshold: {user_profile['night_motion_threshold']}% (child may wake up)")
    
    print(f"\n=== Motion Statistics ===")
    print(f"Total motions: {motion_stats['total_motions']}")
    print(f"Day motions: {motion_stats['day_motions']} ({motion_stats['day_percentage']:.1f}%)")
    print(f"Night motions: {motion_stats['night_motions']} ({motion_stats['night_percentage']:.1f}%)")
    print(f"Empty house motions: {motion_stats['empty_house_motions']} ({motion_stats['empty_house_percentage']:.1f}%)")
    
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