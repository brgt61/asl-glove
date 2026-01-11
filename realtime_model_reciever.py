import socket
import torch 
import scipy.signal as signal
import numpy as np 
import time 
import threading
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
import json
import csv
import os 

# Sensor / Filter config
NUM_SENSORS = 4
REFERENCE_SENSOR = 0  # index of reference sensor

HOST = "10.29.54.16" #10.0.0.118"
PORT = 5050

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
print("Server listening...")

conn, addr = s.accept()
print("Connected by", addr)

# Current buffer sizes
print("Recv buffer size:", conn.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
print("Send buffer size:", conn.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))
# Recv buffer size: 65536
# Send buffer size: 65536
# Real ESP32 limit to send 5760 bytes

# Set buffer size (example: 64 KB is typical limit)
# conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
# conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

# --- shared flag to stop threads ---
running = True
buffer = ""

output_file = "TRIALS/3.csv"
            
# Write header once
if not os.path.exists(output_file):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "sensor_index", "ax", "ay", "az", "gx", "gy", "gz"])


def receiver():
    """Receive and process data from the server."""
    global running
    
    # Open once, append
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)

        
        while running:
            try:
                # Pre-allocate arrays
                
                data = conn.recv(65536)
                ts = time.time()
        
                if not data:
                    print("Server closed connection.")
                    running = False
                    break
                # print("Received data chunk")

                received_json = data.decode()
                line = received_json.split("\n")[0]
                sensor_data_int = json.loads(line)

                LL = len(sensor_data_int[0]["ax"])

                ### for csv file
                for i, s in enumerate(sensor_data_int):
                    LL = len(s["ax"])
                    for t in range(LL):
                        row = [
                            s["time"][t],
                            i,  # sensor index
                            s["ax"][t],
                            s["ay"][t],
                            s["az"][t],
                            s["gx"][t],
                            s["gy"][t],
                            s["gz"][t]
                        ]
                        writer.writerow(row)
                f.flush()  # ensure data is written immediately

                print("Recieving, processing & filtering done in ", round(time.time()-ts, 3), "s")
                ts = time.time()

            except (ConnectionResetError, OSError):
                print("Connection error.")
                running = False
                break
            except Exception as e:
                print("Error:", e)
                # time.sleep(0.1)

    print("Receiver thread exiting.")

# --- start receiver thread ---
thread = threading.Thread(target=receiver, daemon=True)
thread.start()

# --- main thread just waits for Ctrl+C ---
try:
    while running:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nCtrl+C pressed — shutting down...")
    running = False
    conn.close()
    thread.join()

print("Exited cleanly.")

