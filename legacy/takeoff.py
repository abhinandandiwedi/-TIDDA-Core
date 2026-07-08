import collections
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping

from dronekit import connect, VehicleMode
import time

print("🚀 Connecting to TIDDA SITL...")
vehicle = connect('tcp:127.0.0.1:5760', wait_ready=True)

print("💤 Letting SITL drone fully boot up (Waiting 40 seconds)...")
time.sleep(40)

print("🔧 Disabling internal Pre-Arm Safety Checks...")
vehicle.parameters['ARMING_CHECK'] = 0
time.sleep(1)

print("⏳ Waiting for Drone to Initialize & Get GPS Lock...")
while not vehicle.is_armable:
    print("   -> GPS calibrating... Please wait (Takes 10-20 seconds)")
    time.sleep(2)

print("✅ Drone is armable! Proceeding with blind commands...")

print("⏳ Drone says it's armable, but waiting 15 seconds for final GROUND START boot...")
time.sleep(15)

print("🛡️ Setting GUIDED mode blindly...")
vehicle.mode = VehicleMode("GUIDED")
time.sleep(2)

print("⚡ Arming blindly...")
vehicle.armed = True
time.sleep(2)

print("🚁 Taking off to 15 meters!")
vehicle.simple_takeoff(15)

time.sleep(2)
print("✅ Takeoff command sent! Drone is climbing.")
print("👉 FAST: Press Ctrl+C here and run 'python mavlink_bridge.py' again!")