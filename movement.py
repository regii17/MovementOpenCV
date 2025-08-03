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