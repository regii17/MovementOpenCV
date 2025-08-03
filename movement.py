import cv2
import numpy as np
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time

HSV_MIN = (77, 40, 0)    # Range HSV untuk deteksi objek (biru)
HSV_MAX = (101, 255, 255)
TARGET_X = 0.5           # Pusat frame horizontal (0-1)
TARGET_Y = 0.5           # Pusat frame vertikal (0-1)
TOLERANCE = 10           # Toleransi piksel untuk "OBJECT OK"
MOVEMENT_GAIN = 0.001    # Gain kontrol gerakan
WAYPOINT_ALTITUDE = 2    # Ketinggian waypoint (meter)
DESCENT_ALTITUDE = 0.2   # Langkah penurunan ketinggian (meter)
LANDING_ALTITUDE = 0.25  # Ketinggian landing (meter)
WAYPOINT_ACCEPT_RADIUS = 1.0  # Jari-jari toleransi waypoint (meter)

TARGET_WAYPOINT = LocationGlobalRelative(-6.175392, 106.827153, WAYPOINT_ALTITUDE)

def check_waypoint_reached(vehicle, target_location, acceptance_radius=1.0):
    current_location = vehicle.location.global_relative_frame
    distance = get_distance_meters(current_location, target_location)
    return distance <= acceptance_radius

def wait_until_waypoint_reached(vehicle, target_location, timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_waypoint_reached(vehicle, target_location):
            return True
        time.sleep(1)
    return False

def arm_and_takeoff(vehicle, altitude):
    print("Arming motors...")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    
    while not vehicle.armed:
        print("Menunggu arming...")
        time.sleep(1)
    
    print(f"Takeoff ke {altitude} meter!")
    vehicle.simple_takeoff(altitude)
    
    while True:
        current_alt = vehicle.rangefinder.distance
        if current_alt >= altitude * 0.95:
            break
        print(f"Ketinggian: {current_alt:.2f} meter")
        time.sleep(1)

def adjust_position_based_on_object(vehicle, dx, dy):
    velocity_x = -dy * MOVEMENT_GAIN  # Koreksi sumbu Y (timur/barat)
    velocity_y = -dx * MOVEMENT_GAIN  # Koreksi sumbu X (utara/selatan)
    print(f"Koreksi posisi: X={velocity_y:.2f} m/s, Y={velocity_x:.2f} m/s")
    send_ned_velocity(vehicle, velocity_y, velocity_x, 0, 1)

def landing_procedure(vehicle):
    print("Memulai prosedur landing...")
    vehicle.mode = VehicleMode("LAND")
    while vehicle.armed:
        print(f"Ketinggian saat ini: {vehicle.rangefinder.distance:.2f} meter")
        time.sleep(1)
    print("Drone telah mendarat!")

def get_distance_meters(location1, location2):
    dlat = location2.lat - location1.lat
    dlong = location2.lon - location1.lon
    return np.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5
def send_ned_velocity(vehicle, velocity_x, velocity_y, velocity_z, duration):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, 0, 0, # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z, # x, y, z velocity in m/s
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink) 
    
    for x in range(0, duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)

def goto_position(vehicle, lat, lon, alt):
    location = LocationGlobalRelative(lat, lon, alt)
    vehicle.simple_goto(location)
    
    while True:
        current_location = vehicle.location.global_relative_frame
        remaining_distance = get_distance_meters(current_location, location)
        if remaining_distance <= 1.0:  # 1 meter threshold
            break
        time.sleep(1)
def draw_keypoints(image, keypoints, line_color=(0,0,255)):
    return cv2.drawKeypoints(image, keypoints, np.array([]), line_color, 
                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
def get_position_difference(keyPoint, target_x, target_y):
    blob_x = int(keyPoint.pt[0])
    blob_y = int(keyPoint.pt[1])
    return blob_x - target_x, blob_y - target_y

def apply_search_window(image, window_adim=[0.0, 0.0, 1.0, 1.0]):
    rows = image.shape[0]
    cols = image.shape[1]
    x_min_px = int(cols * window_adim[0])
    y_min_px = int(rows * window_adim[1])
    x_max_px = int(cols * window_adim[2])
    y_max_px = int(rows * window_adim[3])    
    
    mask = np.zeros(image.shape, np.uint8)
    mask[y_min_px:y_max_px, x_min_px:x_max_px] = image[y_min_px:y_max_px, x_min_px:x_max_px]   
    return mask

def blob_detect(image, hsv_min, hsv_max, blur=0, blob_params=None, search_window=None):
    if blur > 0: 
        image = cv2.blur(image, (blur, blur))
        
    if search_window is None: 
        search_window = [0.0, 0.0, 1.0, 1.0]
    
    # Convert to HSV and apply threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    
    # Morphological operations
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)
    
    # Apply search window
    mask = apply_search_window(mask, search_window)
    
    # Create blob detector
    if blob_params is None:
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 100
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 20000
        params.filterByCircularity = True
        params.minCircularity = 0.1
        params.filterByConvexity = True
        params.minConvexity = 0.5
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
    else:
        params = blob_params     

    detector = cv2.SimpleBlobDetector_create(params)
    reversemask = 255 - mask
    keypoints = detector.detect(reversemask)
    
    return keypoints, mask

def draw_target(image, target_x=0.5, target_y=0.5, size=0.1, line=2):
    rows = image.shape[0]
    cols = image.shape[1]
    
    center_x = int(cols * target_x)
    center_y = int(rows * target_y)
    line_length = int(min(rows, cols) * size)
    
    cv2.line(image, (center_x - line_length, center_y), (center_x + line_length, center_y), (0, 255, 255), line)
    cv2.line(image, (center_x, center_y - line_length), (center_x, center_y + line_length), (0, 255, 255), line)
    cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
    
    return image, center_x, center_y


def main():
    print("\nMenghubungkan ke drone...")
    vehicle = connect('/dev/ttyS4', wait_ready=True, baud=921600)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Kamera tidak dapat diakses!")
        vehicle.close()
        exit()

    try:
        print("\n=== FASE TAKEOFF ===")
        arm_and_takeoff(vehicle, WAYPOINT_ALTITUDE)
        
        print(f"\n=== FASE MENUJU WAYPOINT ===")
        print(f"Target: Lat={TARGET_WAYPOINT.lat}, Lon={TARGET_WAYPOINT.lon}")
        vehicle.simple_goto(TARGET_WAYPOINT)
        
        if not wait_until_waypoint_reached(vehicle, TARGET_WAYPOINT):
            print("Peringatan: Drone tidak mencapai waypoint dalam waktu yang ditentukan!")
          
        print("\n=== FASE DETEKSI OBJEK ===")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Gagal membaca frame kamera!")
                break
            
            frame, target_x, target_y = draw_target(frame, TARGET_X, TARGET_Y)
            keypoints, mask = blob_detect(frame, HSV_MIN, HSV_MAX, blur=3)
            
            cv2.imshow("Detection Mask", mask)
            if keypoints:
                frame = draw_keypoints(frame, keypoints)
                dx, dy = get_position_difference(keypoints[0], target_x, target_y)
                cv2.putText(frame, f"DX: {dx} DY: {dy}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if abs(dx) <= TOLERANCE and abs(dy) <= TOLERANCE:
                    cv2.putText(frame, "OBJECT OK", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    current_alt = vehicle.rangefinder.distance
                    if current_alt > LANDING_ALTITUDE:
                        new_alt = max(current_alt - DESCENT_ALTITUDE, LANDING_ALTITUDE)
                        print(f"Objek terdeteksi! Menurunkan ke {new_alt:.2f} meter")
                        vehicle.simple_goto(LocationGlobalRelative(
                            vehicle.location.global_relative_frame.lat,
                            vehicle.location.global_relative_frame.lon,
                            new_alt
                        ))
                    else:
                        print("Ketinggian landing tercapai!")
                        break
                else:
                    adjust_position_based_on_object(vehicle, dx, dy)
            
            cv2.imshow("Live View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print("\n=== FASE LANDING ===")
        landing_procedure(vehicle)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if vehicle.armed:
            vehicle.mode = VehicleMode("LAND")
        vehicle.close()
        print("Misi selesai!")

if __name__ == "__main__":
    main()