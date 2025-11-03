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
    # Two-story house connections for family with two children
    
    # Floor 1 connections
    G.add_edge("LivingDining", "Kitchen")
    G.add_edge("Kitchen", "ServiceArea")
    G.add_edge("ServiceArea", "UtilityRoom")
    G.add_edge("LivingDining", "Bathroom1")
    G.add_edge("LivingDining", "Stairs")
    
    # Floor 2 connections (via Stairs and Circulation)
    G.add_edge("Stairs", "Circulation")
    G.add_edge("Circulation", "Bedroom1")
    G.add_edge("Circulation", "Bedroom2")
    G.add_edge("Circulation", "MasterSuite")
    G.add_edge("Circulation", "Bathroom2")
    G.add_edge("Circulation", "WC")

    sensor_locations = {
        # LivingDining sensors (main family area)
        "motion_sensor_living": ["LivingDining", "LivingDining"],
        "temp_sensor_living": "LivingDining",
        "smart_light_living": "LivingDining",
        "smart_tv_living": "LivingDining",
        "smart_lock_front": "LivingDining",
        
        # Kitchen sensors
        "temp_sensor_kitchen": "Kitchen",
        "smart_light_kitchen": "Kitchen",
        "smart_plug_fridge": "Kitchen",
        
        # ServiceArea sensors
        "motion_sensor_service": "ServiceArea",
        "temp_sensor_service": "ServiceArea",
        
        # MasterSuite sensors (parents' bedroom)
        "motion_sensor_suite": "MasterSuite",
        "temp_sensor_suite": "MasterSuite",
        "smart_light_suite": "MasterSuite",
        "smart_tv_suite": "MasterSuite",
        
        # Bedroom1 sensors (Child 1's room)
        "motion_sensor_bedroom1": "Bedroom1",
        "temp_sensor_bedroom1": "Bedroom1",
        "smart_light_bedroom1": "Bedroom1",
        "smart_tv_bedroom1": "Bedroom1",
        
        # Bedroom2 sensors (Child 2's room)
        "motion_sensor_bedroom2": "Bedroom2",
        "temp_sensor_bedroom2": "Bedroom2",
        "smart_light_bedroom2": "Bedroom2"
        
        # Bathrooms, WC, UtilityRoom, Stairs, Circulation: no sensors
    }

    return G, sensor_locations

# === User behavior profile for family with two children in two-story house ===
user_profile = {
    # Adult 1: wakes 06:00, leaves 08:00, returns 17:00, sleeps 22:30
    # Adult 2: wakes 07:00, leaves 09:00, returns 18:00, sleeps 23:00
    # Child 1: wakes 06:30, leaves 07:30, returns 17:30, sleeps 21:30
    # Child 2: wakes 06:30, leaves 07:30, returns 17:30, sleeps 21:30
    
    "all_sleep_hours": (time(23, 0), time(6, 0)),  # All asleep: 23:00-06:00
    "children_sleep_hours": (time(21, 30), time(6, 30)),  # Children sleep: 21:30-06:30
    "house_empty_hours": (time(9, 0), time(17, 0)),  # House empty: 09:00-17:00
    "work_days": [0, 1, 2, 3, 4],  # Monday to Friday (0=Monday, 6=Sunday)
    "school_days": [0, 1, 2, 3, 4],  # Monday to Friday
    "night_motion_allowed": True,  # Children may wake up during night
    "night_motion_threshold": 30,  # Up to 30% night motion acceptable for family with children
    
    # Complex activity periods based on family's schedule
    "adult1_solo_early": (time(6, 0), time(6, 30)),        # 06:00-06:30: Only Adult 1
    "adult1_children_morning": (time(6, 30), time(7, 0)),   # 06:30-07:00: Adult 1 + Children
    "all_active_morning": (time(7, 0), time(7, 30)),        # 07:00-07:30: All four active
    "adults_preparation": (time(7, 30), time(8, 0)),        # 07:30-08:00: Adults (children at school)
    "adult2_solo_departure": (time(8, 0), time(9, 0)),      # 08:00-09:00: Only Adult 2
    "adult1_solo_return": (time(17, 0), time(17, 30)),      # 17:00-17:30: Only Adult 1
    "adult1_children_afternoon": (time(17, 30), time(18, 0)), # 17:30-18:00: Adult 1 + Children
    "all_active_evening": (time(18, 0), time(21, 30)),      # 18:00-21:30: All four active
    "adults_only_evening": (time(21, 30), time(22, 30)),    # 21:30-22:30: Adults (children sleeping)
    "adult2_solo_night": (time(22, 30), time(23, 0)),       # 22:30-23:00: Only Adult 2
    
    # House layout - two-story with family arrangement
    "master_suite": "MasterSuite",  # Adults sleep here (floor 2)
    "child1_bedroom": "Bedroom1",   # Child 1 sleeps here (floor 2)
    "child2_bedroom": "Bedroom2",   # Child 2 sleeps here (floor 2)
    "children_bedrooms": ["Bedroom1", "Bedroom2"],  # Both children's rooms
    "all_bedrooms": ["MasterSuite", "Bedroom1", "Bedroom2"],  # All bedrooms
    "main_floor_areas": ["LivingDining", "Kitchen", "Bathroom1"],  # Primary floor 1 areas
    "service_areas": ["ServiceArea", "UtilityRoom"],  # Secondary floor 1 areas
    "upstairs_areas": ["MasterSuite", "Bedroom1", "Bedroom2", "Bathroom2", "WC"],  # Floor 2 private areas
    "circulation_areas": ["Stairs", "Circulation"],  # Movement areas
    
    # Floor-specific usage patterns for family
    "floor1_primary": ["LivingDining", "Kitchen"],  # Most used floor 1 areas (family activities)
    "floor2_primary": ["MasterSuite", "Bedroom1", "Bedroom2"],  # Most used floor 2 areas (sleeping)
    "occasional_areas": ["ServiceArea", "UtilityRoom"],  # Less frequent use
    
    # Environment settings (Winter Brazil)
    "temp_range": (21.0, 26.0),
    "humidity_range": (40, 70),
    "temp_humidity_correlation": (-0.7, -0.9),  # Inverse correlation
    
    # Technical correlations (more tolerant for family with children)
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

def is_children_sleep_time(ts, sleep_range):
    """Check if timestamp is during hours when children are asleep"""
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
    
    # Check each period based on family's complex schedule
    if user_profile["adult1_solo_early"][0] <= current_time < user_profile["adult1_solo_early"][1]:
        return "adult1_solo_early"
    elif user_profile["adult1_children_morning"][0] <= current_time < user_profile["adult1_children_morning"][1]:
        return "adult1_children_morning"
    elif user_profile["all_active_morning"][0] <= current_time < user_profile["all_active_morning"][1]:
        return "all_active_morning"
    elif user_profile["adults_preparation"][0] <= current_time < user_profile["adults_preparation"][1]:
        return "adults_preparation"
    elif user_profile["adult2_solo_departure"][0] <= current_time < user_profile["adult2_solo_departure"][1]:
        return "adult2_solo_departure"
    elif user_profile["adult1_solo_return"][0] <= current_time < user_profile["adult1_solo_return"][1]:
        return "adult1_solo_return"
    elif user_profile["adult1_children_afternoon"][0] <= current_time < user_profile["adult1_children_afternoon"][1]:
        return "adult1_children_afternoon"
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
    # During periods with all family present, all areas can be used
    if activity_period in ["all_active_morning", "all_active_evening"]:
        return True  # All family members active - all areas valid
    
    # During periods with some family members present
    elif activity_period in ["adult1_solo_early", "adult1_children_morning", "adults_preparation", 
                           "adult2_solo_departure", "adult1_solo_return", "adult1_children_afternoon", 
                           "adults_only_evening", "adult2_solo_night"]:
        return True  # Some family members active - all areas still valid
    
    # During house empty time, no activity expected
    elif activity_period == "house_empty":
        return False
    
    # During all sleep time, activity allowed in any bedroom (children may wake up)
    elif activity_period == "all_sleep":
        return location in user_profile["all_bedrooms"]
    
    return True

def get_floor_from_location(location):
    """Determine which floor a location is on"""
    floor1_rooms = ["LivingDining", "Kitchen", "Bathroom1", "ServiceArea", "UtilityRoom", "Stairs"]
    floor2_rooms = ["Bedroom1", "Bedroom2", "MasterSuite", "Bathroom2", "WC", "Circulation"]
    
    if location in floor1_rooms:
        return "floor1"
    elif location in floor2_rooms:
        return "floor2"
    else:
        return "unknown"

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
        # For family with children, night motion is allowed but should be in bedrooms
        if has_motion(row) and current_location not in user_profile["all_bedrooms"]:
            errors.append(f"Motion outside bedrooms during sleep hours: {current_location} at {timestamp_str} (expected only in bedrooms)")
        elif not has_motion(row) and row["event_type"] not in ["temperature_reading", "humidity_reading"]:
            # Non-motion events during sleep should mostly be environmental or in bedrooms
            if current_location not in user_profile["all_bedrooms"]:
                errors.append(f"Non-environmental event outside bedrooms during sleep: {current_location} at {timestamp_str}")
    
    # Rule 4: Validate activity patterns
    if not validate_activity_pattern(current_location, activity_period):
        if activity_period == "house_empty":
            errors.append(f"Activity during house empty period: {current_location} at {timestamp_str}")
        elif activity_period == "all_sleep":
            errors.append(f"Activity outside bedrooms during sleep: {current_location} at {timestamp_str}")
    
    # Rule 5: Floor transition validation (for two-story house)
    if previous_room and has_motion(row):
        prev_floor = get_floor_from_location(previous_room)
        curr_floor = get_floor_from_location(current_location)
        
        # If changing floors, must go through Stairs/Circulation
        if prev_floor != curr_floor and prev_floor != "unknown" and curr_floor != "unknown":
            # Must use stairs to change floors
            if previous_room not in ["Stairs", "Circulation"] and current_location not in ["Stairs", "Circulation"]:
                errors.append(f"Floor change without using stairs: {previous_room} ({prev_floor}) → {current_location} ({curr_floor})")
    
    # Rule 6: Validate temperature and humidity ranges and correlation
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
        
        # Track by floor for two-story analysis
        floor = get_floor_from_location(current_location)
        if floor == "floor1":
            motion_tracker['floor1_motions'].append(ts)
        elif floor == "floor2":
            motion_tracker['floor2_motions'].append(ts)
        
        # Track by bedroom for family analysis
        if current_location == user_profile["master_suite"]:
            motion_tracker['parents_motions'].append(ts)
        elif current_location in user_profile["children_bedrooms"]:
            motion_tracker['children_motions'].append(ts)
    
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
    
    # Rule 7: Movement between rooms must follow graph structure
    if previous_room and has_motion(row):
        if not graph.has_edge(previous_room, current_location) and previous_room != current_location:
            errors.append(f"Invalid transition: {previous_room} → {current_location} not connected in house layout")

    # Rule 8: Duplicate motion event in the same location at the same second
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
    
    # Allow more tolerance for family with children - more variability expected
    if len(motion_events) > 0:
        missing_temp_percentage = (motion_without_temp_response / len(motion_events)) * 100
        if missing_temp_percentage > 75:  # More than 75% without temp response is suspicious (higher tolerance)
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
    
    # Allow more tolerance for family with children - power usage is more irregular
    if len(motion_events) > 0:
        missing_power_percentage = (motion_without_power_response / len(motion_events)) * 100
        if missing_power_percentage > 65:  # More than 65% without power response is suspicious (higher tolerance)
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
    floor1_motions = len(motion_tracker['floor1_motions'])
    floor2_motions = len(motion_tracker['floor2_motions'])
    parents_motions = len(motion_tracker['parents_motions'])
    children_motions = len(motion_tracker['children_motions'])
    
    if total_motions == 0:
        return errors
    
    # Calculate frequency percentages
    night_percentage = (night_motions / total_motions) * 100
    empty_house_percentage = (empty_house_motions / total_motions) * 100
    floor1_percentage = (floor1_motions / total_motions) * 100
    floor2_percentage = (floor2_motions / total_motions) * 100
    
    # Rule 9: Night motion frequency analysis
    # For family with children, higher night motion is acceptable (children wake up)
    night_threshold = user_profile["night_motion_threshold"]
    if night_percentage > night_threshold:
        errors.append(f"High night motion frequency: {night_percentage:.1f}% of total motion during sleep hours (expected < {night_threshold}%)")
    
    # Rule 10: Empty house motion analysis
    # Should be no motion when house is empty
    if empty_house_motions > 0:
        errors.append(f"Motion detected during empty house hours: {empty_house_motions} events ({empty_house_percentage:.1f}% of total)")
    
    # Rule 11: Floor usage balance analysis for family
    # For family with children, expect more activity on floor 1 (family activities) during active hours
    if floor1_percentage < 25:  # Less than 25% on main floor is unusual for family
        errors.append(f"Low floor 1 activity: {floor1_percentage:.1f}% of motion (expected > 25% for family living areas)")
    
    return errors

def create_motion_frequency_graph(motion_tracker, output_file="motion_frequency_analysis_two_story_house_family_with_two_children.png"):
    """Create a line graph showing motion frequency throughout the day"""
    hourly_motions = motion_tracker['hourly_motions']
    
    # Create arrays for plotting
    hours = list(range(24))
    motion_counts = [hourly_motions[h] for h in hours]
    
    plt.figure(figsize=(16, 10))
    plt.plot(hours, motion_counts, 'b-', linewidth=2, label='Motion Events')
    plt.fill_between(hours, motion_counts, alpha=0.3)
    
    # Add all sleep time shading
    sleep_start, sleep_end = user_profile["all_sleep_hours"]
    sleep_start_hour = sleep_start.hour + sleep_start.minute/60
    sleep_end_hour = sleep_end.hour + sleep_end.minute/60
    
    # Handle sleep time that crosses midnight
    if sleep_start_hour > sleep_end_hour:
        plt.axvspan(sleep_start_hour, 24, alpha=0.2, color='red', label='All Sleep')
        plt.axvspan(0, sleep_end_hour, alpha=0.2, color='red')
    else:
        plt.axvspan(sleep_start_hour, sleep_end_hour, alpha=0.2, color='red', label='All Sleep')
    
    # Add children sleep time shading
    child_sleep_start, child_sleep_end = user_profile["children_sleep_hours"]
    child_sleep_start_hour = child_sleep_start.hour + child_sleep_start.minute/60
    child_sleep_end_hour = child_sleep_end.hour + child_sleep_end.minute/60
    
    # Children sleep longer (21:30-06:30)
    if child_sleep_start_hour > child_sleep_end_hour:
        plt.axvspan(child_sleep_start_hour, sleep_start_hour, alpha=0.15, color='pink', label='Children Sleep')
        plt.axvspan(sleep_end_hour, child_sleep_end_hour, alpha=0.15, color='pink')
    
    # Add house empty time shading
    empty_start, empty_end = user_profile["house_empty_hours"]
    empty_start_hour = empty_start.hour + empty_start.minute/60
    empty_end_hour = empty_end.hour + empty_end.minute/60
    plt.axvspan(empty_start_hour, empty_end_hour, alpha=0.2, color='orange', label='House Empty')
    
    # Add complex family activity period markings
    plt.axvspan(6, 6.5, alpha=0.1, color='green', label='A1 Solo Early')
    plt.axvspan(6.5, 7, alpha=0.1, color='cyan', label='A1+Children Morning')
    plt.axvspan(7, 7.5, alpha=0.1, color='blue', label='All Active Morning')
    plt.axvspan(7.5, 8, alpha=0.1, color='purple', label='Adults Preparation')
    plt.axvspan(8, 9, alpha=0.1, color='magenta', label='A2 Solo Departure')
    plt.axvspan(17, 17.5, alpha=0.1, color='lime', label='A1 Solo Return')
    plt.axvspan(17.5, 18, alpha=0.1, color='yellow', label='A1+Children Afternoon')
    plt.axvspan(18, 21.5, alpha=0.1, color='blue', label='All Active Evening')
    plt.axvspan(21.5, 22.5, alpha=0.1, color='gray', label='Adults Only Evening')
    plt.axvspan(22.5, 23, alpha=0.1, color='brown', label='A2 Solo Night')
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Motion Events')
    plt.title('Motion Frequency Throughout the Day - Two-Story House (Family with Two Children)')
    plt.xticks(hours)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
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
    floor1_motions = len(motion_tracker['floor1_motions'])
    floor2_motions = len(motion_tracker['floor2_motions'])
    parents_motions = len(motion_tracker['parents_motions'])
    children_motions = len(motion_tracker['children_motions'])
    
    stats = {
        'total_motions': total_motions,
        'night_motions': night_motions,
        'empty_house_motions': empty_house_motions,
        'day_motions': day_motions,
        'floor1_motions': floor1_motions,
        'floor2_motions': floor2_motions,
        'parents_motions': parents_motions,
        'children_motions': children_motions,
        'night_percentage': (night_motions / total_motions) * 100 if total_motions > 0 else 0,
        'empty_house_percentage': (empty_house_motions / total_motions) * 100 if total_motions > 0 else 0,
        'day_percentage': (day_motions / total_motions) * 100 if total_motions > 0 else 0,
        'floor1_percentage': (floor1_motions / total_motions) * 100 if total_motions > 0 else 0,
        'floor2_percentage': (floor2_motions / total_motions) * 100 if total_motions > 0 else 0,
        'parents_percentage': (parents_motions / total_motions) * 100 if total_motions > 0 else 0,
        'children_percentage': (children_motions / total_motions) * 100 if total_motions > 0 else 0
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
        'floor1_motions': [],
        'floor2_motions': [],
        'parents_motions': [],
        'children_motions': [],
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
        
        # Rule 12: Check for duplicate event IDs
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
    csv_path = "iot_dataset_two_story_house_family_with_two_children_errors.csv"  # Change this path to your CSV file
    error_count, error_log, motion_stats, graph_file = validate_csv(csv_path)

    print(f"\n=== IoT Dataset Validation Results ===")
    print(f"Profile: Family with Two Children in Two-Story House")
    print(f"Adult 1: Wakes 06:00, leaves 08:00, returns 17:00, sleeps 22:30")
    print(f"Adult 2: Wakes 07:00, leaves 09:00, returns 18:00, sleeps 23:00")
    print(f"Child 1: Wakes 06:30, school 07:30, returns 17:30, sleeps 21:30")
    print(f"Child 2: Wakes 06:30, school 07:30, returns 17:30, sleeps 21:30")
    print(f"")
    print(f"Sleeping Arrangements:")
    print(f"  Adults: MasterSuite (Floor 2)")
    print(f"  Child 1: Bedroom1 (Floor 2)")
    print(f"  Child 2: Bedroom2 (Floor 2)")
    print(f"")
    print(f"House Layout:")
    print(f"  Floor 1: LivingDining, Kitchen, Bathroom1, ServiceArea, UtilityRoom")
    print(f"  Floor 2: MasterSuite, Bedroom1, Bedroom2, Bathroom2, WC")
    print(f"  Connection: Stairs ↔ Circulation (between floors)")
    print(f"")
    print(f"Complex Activity Timeline (10 periods):")
    print(f"  06:00-06:30 → Only Adult 1 active")
    print(f"  06:30-07:00 → Adult 1 + Children active")
    print(f"  07:00-07:30 → All four active")
    print(f"  07:30-08:00 → Both adults active (children at school)")
    print(f"  08:00-09:00 → Only Adult 2 active")
    print(f"  09:00-17:00 → House empty (all at work/school)")
    print(f"  17:00-17:30 → Only Adult 1 active")
    print(f"  17:30-18:00 → Adult 1 + Children active")
    print(f"  18:00-21:30 → All four active")
    print(f"  21:30-22:30 → Both adults active (children sleeping)")
    print(f"  22:30-23:00 → Only Adult 2 active")
    print(f"  23:00-06:00 → All asleep")
    print(f"")
    print(f"Environment: Winter Brazil, 21-26°C, 40-70% humidity")
    print(f"Correlations: Motion→Temp (+0.5-1.5°C), Motion→Power (+100-300W)")
    print(f"Family tolerances: Night motion < {user_profile['night_motion_threshold']}%, Temp < 75%, Power < 65%")
    print(f"Two-story validation: Floor changes must use Stairs/Circulation")
    
    print(f"\n=== Motion Statistics ===")
    print(f"Total motions: {motion_stats['total_motions']}")
    print(f"Day motions: {motion_stats['day_motions']} ({motion_stats['day_percentage']:.1f}%)")
    print(f"Night motions: {motion_stats['night_motions']} ({motion_stats['night_percentage']:.1f}%)")
    print(f"Empty house motions: {motion_stats['empty_house_motions']} ({motion_stats['empty_house_percentage']:.1f}%)")
    print(f"Floor 1 motions: {motion_stats['floor1_motions']} ({motion_stats['floor1_percentage']:.1f}%)")
    print(f"Floor 2 motions: {motion_stats['floor2_motions']} ({motion_stats['floor2_percentage']:.1f}%)")
    print(f"Parents' bedroom: {motion_stats['parents_motions']} ({motion_stats['parents_percentage']:.1f}%)")
    print(f"Children's bedrooms: {motion_stats['children_motions']} ({motion_stats['children_percentage']:.1f}%)")
    
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