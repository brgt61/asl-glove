import time
import math
import random
import adafruit_mpu6050
import wifi
import socketpool
import json
import board
import busio
import adafruit_tca9548a
from adafruit_bus_device.i2c_device import I2CDevice
import struct


i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
# i2c = busio.I2C(board.SCL, board.SDA)
tca = adafruit_tca9548a.TCA9548A(i2c)

# --- Sensor / Filter config ---
NUM_SENSORS = 4
# LIMIT 0.075s (incl moments when sending, otherwise 0.075 because take a while to sample 8 sensors)
# FS = 14 # Hz ## NOT ACTUALLY USED NOW, JUST DO WHAT WE CAN, IT'S APPROX THIS
# SCALE = 32767
SEND_INTERVAL = 0.5 ## CHOSEN TO BE NICE
# CHUNK_SIZE = int(SEND_INTERVAL * FS) # as is numbers wise this works out ## NO LONGER USED


print("Connecting sensors")

mpu_list = []
# Assign each sensor to a channel
for ch in range(NUM_SENSORS):
    print('hello')
    try: 
        mpu_list.append(adafruit_mpu6050.MPU6050(tca[ch]))
    except: 
        print(f'No sensor to append on channel: {ch}')

print("Sensors connected")
# mpu_list = mpu_list*NUM_SENSORS ### JUST FOR TESTING W 1 MPU CONNECTED
print(len(mpu_list))

# Helper: Read sensors
def update_sensors(sensor_data_, start_time):
    for i in range(NUM_SENSORS):
        ax, ay, az = mpu_list[i].acceleration  # m/s²
        gx, gy, gz = mpu_list[i].gyro  # rad/s
        
        # sensor_data_[i]["ax"].append(int(ax * SCALE))
        # sensor_data_[i]["ay"].append(int(ay * SCALE))
        # sensor_data_[i]["az"].append(int(az * SCALE))
        # sensor_data_[i]["gx"].append(int(gx * SCALE))
        # sensor_data_[i]["gy"].append(int(gy * SCALE))
        # sensor_data_[i]["gz"].append(int(gz * SCALE))
        # sensor_data_[i]["time"].append(int((time.monotonic()-start_time) * SCALE))

        # sensor_data_[i]["ax"].append(int(ax))
        # sensor_data_[i]["ay"].append(int(ay))
        # sensor_data_[i]["az"].append(int(az))
        # sensor_data_[i]["gx"].append(int(gx))
        # sensor_data_[i]["gy"].append(int(gy))
        # sensor_data_[i]["gz"].append(int(gz))
        # sensor_data_[i]["time"].append(int((time.monotonic()-start_time)))

        sensor_data_[i]["ax"].append(ax)
        sensor_data_[i]["ay"].append(ay)
        sensor_data_[i]["az"].append(az)
        sensor_data_[i]["gx"].append(gx)
        sensor_data_[i]["gy"].append(gy)
        sensor_data_[i]["gz"].append(gz)
        sensor_data_[i]["time"].append(time.monotonic()-start_time)


    return sensor_data_



sensor_data = [{"ax": [], "ay": [], "az": [], "gx": [], "gy": [], "gz": [], "time": []} for _ in range(NUM_SENSORS)]


# Wi-Fi credentials
SSID = "asik" #"dihydrogenmonoxide" # 
PASSWORD = "agent10x" #elevencoast" #"

# Server info (your computer)
HOST = "10.29.54.16" #"10.0.0.118"
PORT = 5050

# Connect to Wi-Fi
print("Connecting to Wi-Fi...")
wifi.radio.connect(SSID, PASSWORD)
print("Connected!")

# Create socket pool
pool = socketpool.SocketPool(wifi.radio)

# Connect to server
print("Connecting to server...")
sock = pool.socket(pool.AF_INET, pool.SOCK_STREAM)
sock.connect((HOST, PORT))
print("Connected to server!")

# while True:
#     sensor_data = update_sensors(sensor_data)
#     time.sleep(1/FS)

last_send_time = time.monotonic()
start_time = time.monotonic()

while True:
    # Collect sensor data
    ts = time.monotonic()
    sensor_data = update_sensors(sensor_data, start_time)

    # Send at intervals (non-blocking collection)
    now = time.monotonic()
    # print(now - last_send_time)
    
   
    if now - last_send_time >= SEND_INTERVAL:
        # ts = time.time()
        try:
            # build payload with only last CHUNK_SIZE values per axis
            # latest_data = []
            # for i in range(NUM_SENSORS):
            #     latest_data.append({
            #         "ax": list(sensor_data[i]["ax"][-CHUNK_SIZE:]),
            #         "ay": list(sensor_data[i]["ay"][-CHUNK_SIZE:]),
            #         "az": list(sensor_data[i]["az"][-CHUNK_SIZE:]),
            #         "gx": list(sensor_data[i]["gx"][-CHUNK_SIZE:]),
            #         "gy": list(sensor_data[i]["gy"][-CHUNK_SIZE:]),
            #         "gz": list(sensor_data[i]["gz"][-CHUNK_SIZE:])
            #     })

            latest_data = sensor_data ## send whole thing

            # send once
            encoded = json.dumps(latest_data).encode('utf-8')
            r = sock.send(encoded)

            # clear all stored data (fresh start)
            sensor_data = [{"ax": [], "ay": [], "az": [], "gx": [], "gy": [], "gz": [], "time": []}
                           for _ in range(NUM_SENSORS)]

            last_send_time = now

            if len(encoded) != r: 
                print('SENDING WAS BAD!')
                print('needed to send: ', len(encoded), ' | got: ', r)

        except Exception as e:
            print("Send failed:", e) 
            print('DELETING ALL OLD DATA -- WILL HAVE A GAP IN ARRIVED POINTS')
            # Optionally reconnect here
            sensor_data = [{"ax": [], "ay": [], "az": [], "gx": [], "gy": [], "gz": [], "time": []}
                           for _ in range(NUM_SENSORS)]
            last_send_time = now
        
        ## CHECKED THAT NEGLIGIBLE LOSS IN TIME OVER THE SENDING TIME
        # print(time.time()-ts, ' compare against ', 1/FS) 
    
    # else:
    #     print('normal add')
    # Maintain collection frequency

    elapse = time.monotonic()-ts
    print(elapse, ' freq = ', 1/elapse)
    ### WITH INT SENDING: FREQUNCY mostly 25-28Hz w dips to 21 and 32
    ### SIGNALS ARE SENT W THEIR TIMESTEPS










### TEST WIFI CONNECTION

# # print("Scanning for Wi-Fi networks...")
# # for network in wifi.radio.start_scanning_networks():
# #     print(f"SSID: {network.ssid}, RSSI: {network.rssi}")
# # wifi.radio.stop_scanning_networks()

# # Wi-Fi credentials
# SSID = "dihydrogenmonoxide"
# PASSWORD = "elevencoast"

# # Server info (your computer)
# HOST = "10.0.0.118"  # replace with your PC's IP
# PORT = 5000

# # Connect to Wi-Fi
# print("Connecting to Wi-Fi...")
# wifi.radio.connect(SSID, PASSWORD)
# print("Connected!")

# # Create socket pool
# pool = socketpool.SocketPool(wifi.radio)

# # Connect to server
# print("Connecting to server...")
# sock = pool.socket(pool.AF_INET, pool.SOCK_STREAM)
# sock.connect((HOST, PORT))
# print("Connected to server!")

# while True:
#     # Prepare a long list
#     data = ",".join(str(i) for i in range(8*3*2*50*4)) + "\n"
    
#     try:
#         sock.send(data.encode("utf-8"))
#         print("Sent", len(data.encode("utf-8")), "bytes")
#     except OSError as e:
#         if e.errno == 11:  # EAGAIN
#             print("Buffer busy, retrying...")
#             time.sleep(0.05)  # small pause before retry
#             continue
#         else:
#             raise e

#     # time.sleep(0.25)  # send every x seconds




### RANDOM TESTING 

# for ch in range(8):
#     if tca[ch].try_lock():
#         devices = [hex(x) for x in tca[ch].scan()]
#         tca[ch].unlock()
#         print(f"Channel {ch}: {devices}")


# # --- Wait for lock and scan ---
# while not i2c.try_lock():
#     pass
# print("I2C locked!")
# print("Scan:", [hex(x) for x in i2c.scan()])
# i2c.unlock()