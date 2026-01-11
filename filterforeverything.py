import numpy as np
from scipy.spatial.transform import Rotation as R
from ahrs.filters import Madgwick
import matplotlib.pyplot as plt
import os 
import pandas as pd 

np.random.seed(0)


def wxyz_to_scipy(q):
    return np.array([q[1], q[2], q[3], q[0]])

def rotate_vector(q_wxyz, v):
    """Rotate vector v by quaternion q (wxyz)."""
    r = R.from_quat(wxyz_to_scipy(q_wxyz))
    return r.apply(v)

def relative_q(q1, q2):
    """Quaternion mapping sensor2 → sensor1 frame."""
    # q1, q2 are wxyz
    r1 = R.from_quat(wxyz_to_scipy(q1))
    r2 = R.from_quat(wxyz_to_scipy(q2))
    r_rel = r1.inv() * r2
    q = r_rel.as_quat()  # scipy x y z w
    return np.array([q[3], q[0], q[1], q[2]])  # convert to wxyz

def transform_all_to_sensor1_frame(Q1, Qj, acc_j, gyro_j):
    N = len(Q1)
    acc_ref = np.zeros_like(acc_j)
    gyro_ref = np.zeros_like(gyro_j)
    
    for t in range(N):
        # rotation: body_j → body_1
        q_rel = relative_q(Q1[t], Qj[t])
        
        acc_ref[t]  = rotate_vector(q_rel, acc_j[t])
        gyro_ref[t] = rotate_vector(q_rel, gyro_j[t])
        
    return acc_ref, gyro_ref





# Convert quaternions to Euler for comparison (scipy needs [x,y,z,w])
def q_wxyz_to_euler_deg(q_wxyz):
    q_scipy = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return R.from_quat(q_scipy).as_euler('xyz', degrees=True)

# Remove gravity using estimated orientation Q_est
def remove_gravity_body(acc_data, Q):
    N = acc_data.shape[0]
    lin = np.zeros_like(acc_data)
    for i in range(N):
        w,x,y,z = Q[i]
        q_scipy = np.array([x,y,z,w])
        rot = R.from_quat(q_scipy)
        g_b = rot.inv().apply(g_world)
        lin[i] = acc_data[i] - g_b
    return lin
    
# Path to your folder
folder_path = "SUPPDATA" #"REALTIME" #"DATA"
os.makedirs('ALIGNEDEXTRAS', exist_ok=True)

# Collect all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Load each CSV into a DataFrame and store them in a dictionary
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)

    print('FILE: ', file)
    # if file.split('_')[1] != 'EMPTY': 
    #     continue

    # Create a dictionary: key = sensor_index, value = DataFrame for that sensor
    dfs = {sensor: group for sensor, group in df.groupby("sensor_index")}
    df_sensor0 = dfs[0]
    df_sensor1 = dfs[1]
    df_sensor2 = dfs[2]
    df_sensor3 = dfs[3]
    scale_correction = np.array([1.0 for _ in range(len(dfs))])

    for d in range(len(dfs)): 
        x = dfs[d].loc[:, 'ax':'az'].iloc[0:5]
        scale_correction[d] = 9.80665/np.mean(np.linalg.norm(x, axis=1))
    
    print('Scale correction: ', scale_correction)
    
    quaternions_list = []
    de_gravity_acc = []
    position_parameters = []
    gyros_list = []

    times_list = []


    for d in range(len(dfs)): 
        df_specific = dfs[d]

        gyro = df_specific[['gx','gy','gz']].to_numpy()
        acc  = df_specific[['ax','ay','az']].to_numpy()*scale_correction[d]
        delta_t   = df_specific['time'].to_numpy()
        times_list.append(delta_t)

        # Simulation params
        num = len(acc)
        # dt = 0.01
        t = delta_t #np.arange(num) * dt
        g = 9.80665
        g_world = np.array([0.0, 0.0, g])  # gravity pointing +Z down in world

    
        # Run Madgwick filter using accelerometer (should be reliable because motion is slow)
        madgwick = Madgwick(beta=0.0001)  # modest beta: allow accel to correct
        Q_est = np.zeros((num,4))
        # initialize from accelerometer (compute roll,pitch from first acc sample)
        acc0 = acc[0] / np.linalg.norm(acc[0])
        ax, ay, az = acc0
        roll0 = np.arctan2(ay, az)
        pitch0 = np.arctan2(-ax, np.sqrt(ay*ay + az*az))
        yaw0 = 0.0
        rot0 = R.from_euler('xyz', [roll0, pitch0, yaw0])
        q0_scipy = rot0.as_quat()  # [x,y,z,w]
        Q_est[0] = np.array([q0_scipy[3], q0_scipy[0], q0_scipy[1], q0_scipy[2]])

        for i in range(1, num):
            acc_norm = acc[i] / np.linalg.norm(acc[i])
            gyro_rad = np.radians(gyro[i])
            dt = delta_t[i] - delta_t[i-1]
            Q_est[i] = madgwick.updateIMU(q=Q_est[i-1], gyr=gyro_rad, acc=acc_norm, dt=dt)


        # euler_true = np.array([R.from_quat(Rot_true[i].as_quat()).as_euler('xyz', degrees=True) for i in range(num)])
        euler_est  = np.array([q_wxyz_to_euler_deg(Q_est[i]) for i in range(num)])

        lin_body = remove_gravity_body(acc, Q_est)

### Visualize rotation over time
        # # Find index where x-axis roughly points down: look for ax ~ +g (peak near end)
        # ix_x_down = len(acc)-1 #np.argmin(np.abs(acc[:,0] - g))

        # # Print a short summary and a small window like you asked
        # def print_window(ix, label, window=5):
        #     start = max(0, ix - window)
        #     end = min(num, ix + window + 1)
        #     print(f"\nWindow around {label} (index {ix}):")
        #     print(" idx |   acc (m/s2) (ax,ay,az)                 | gravity-removed linear_body (m/s2)")
        #     for j in range(start, end):
        #         a = acc[j]
        #         l = lin_body[j]
        #         print(f"{j:4d}: {a[0]:7.3f}, {a[1]:7.3f}, {a[2]:7.3f}  |  {l[0]:7.3f}, {l[1]:7.3f}, {l[2]:7.3f}")

        # print("Index where ax ≈ +g (x-axis down):", ix_x_down)
        # print_window(ix_x_down, "x-down", window=6)

        # # Print final Euler comparison at that index
        # # print("\nTrue Euler (deg) at x-down:", euler_true[ix_x_down])
        # print("Estimated Euler (deg) at x-down:", euler_est[ix_x_down])
        # # print("Error (deg):", euler_est[ix_x_down] - euler_true[ix_x_down])

        # # Plot true vs estimated Euler angles over time
        # fig, axs = plt.subplots(3,1, figsize=(9,7), sharex=True)
        # labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
        # for k, axplt in enumerate(axs):
        #     # axplt.plot(t, euler_true[:,k], label='true')
        #     axplt.plot(t, euler_est[:,k], label='est', linestyle='--')
        #     axplt.set_ylabel(labels[k])
        #     axplt.grid(True)
        # axs[-1].set_xlabel('time (s)')
        # axs[0].legend()
        # plt.suptitle('True vs Estimated Euler angles (degrees) — slow motion with X axis ending up down')
        # plt.tight_layout(rect=[0,0,1,0.96])
        # plt.show()

        # # Also show accelerometer magnitudes to ensure they are near g (since motion is slow)
        # mag_raw = np.linalg.norm(acc, axis=1)
        # print("\nAcceleration magnitude (min,mean,max):", mag_raw.min(), mag_raw.mean(), mag_raw.max())

        # # Return a small summary dict for quick inspection
        # summary = {
        #     "index_x_down": int(ix_x_down),
        #     # "true_euler_at_x_down": euler_true[ix_x_down].tolist(),
        #     "est_euler_at_x_down": euler_est[ix_x_down].tolist(),
        #     "acc_at_x_down": acc[ix_x_down].tolist(),
        #     "lin_body_at_x_down": lin_body[ix_x_down].tolist(),
        #     "acc_magnitude_stats": (float(mag_raw.min()), float(mag_raw.mean()), float(mag_raw.max()))
        # }

        # summary

        # exit()

        quaternions_list.append(Q_est)
        de_gravity_acc.append(lin_body)
        position_parameters.append(euler_est)
        gyros_list.append(gyro)


    # --- Apply to each sensor ---
    acc1_ref, gyro1_ref = de_gravity_acc[0], gyros_list[0]  # sensor1 already in sensor1 frame
    acc_ref_list = [acc1_ref]
    gyro_ref_list = [gyro1_ref]
    for k in range(1, len(gyros_list)):     
        acc2_ref, gyro2_ref = transform_all_to_sensor1_frame(quaternions_list[0], quaternions_list[k], de_gravity_acc[k], gyros_list[k])
        acc_ref_list.append(acc2_ref)
        gyro_ref_list.append(gyro2_ref)


    # Stack columns for the CSV
    data = np.hstack([
        times_list[0].reshape(-1, 1), times_list[1].reshape(-1, 1), times_list[2].reshape(-1, 1), times_list[3].reshape(-1, 1),
        acc_ref_list[0], gyro_ref_list[0],
        acc_ref_list[1], gyro_ref_list[1],
        acc_ref_list[2], gyro_ref_list[2],
        acc_ref_list[3], gyro_ref_list[3]
    ])

    # Column names
    columns = ['times1', 'times2', 'times3', 'times4']
    for i in range(1,5):
        columns += [f"acc{i}_x", f"acc{i}_y", f"acc{i}_z",
                    f"gyro{i}_x", f"gyro{i}_y", f"gyro{i}_z"]

    df = pd.DataFrame(data, columns=columns)

### SAVE HERE
    df.to_csv(f"ALIGNEDEXTRAS/{file}", index=False)
    # df.to_csv(f"ALIGNED/{file}", index=False)
    print(f"CSV saved successfully: {file}")



######## VISUALIZE SENSOR OUTPUTS
    # Color-coded by sensor
    # colors = ["red", "blue", "green", "purple"]
    # labels = ["Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4"]

    # # ---------------------------------------------------------
    # # PLOT 1 — Acceleration
    # # ---------------------------------------------------------
    # plt.figure(figsize=(12, 6))
    # for i, acc in enumerate(acc_ref_list):
    #     plt.plot(acc[:,0], color=colors[i], alpha=0.8, label=f"{labels[i]} ax")
    #     plt.plot(acc[:,1], color=colors[i], alpha=0.5, linestyle="--", label=f"{labels[i]} ay")
    #     plt.plot(acc[:,2], color=colors[i], alpha=0.3, linestyle=":", label=f"{labels[i]} az")

    # plt.title("Acceleration (expressed in Sensor 1 frame)")
    # plt.xlabel("Time index")
    # plt.ylabel("Acceleration (m/s²)")
    # plt.legend(ncol=3)
    # plt.grid(True)
    # plt.tight_layout()

    # # ---------------------------------------------------------
    # # PLOT 2 — Gyroscope
    # # ---------------------------------------------------------
    # plt.figure(figsize=(12, 6))
    # for i, gyro in enumerate(gyro_ref_list):
    #     plt.plot(gyro[:,0], color=colors[i], alpha=0.8, label=f"{labels[i]} gx")
    #     plt.plot(gyro[:,1], color=colors[i], alpha=0.5, linestyle="--", label=f"{labels[i]} gy")
    #     plt.plot(gyro[:,2], color=colors[i], alpha=0.3, linestyle=":", label=f"{labels[i]} gz")

    # plt.title("Gyroscope (expressed in Sensor 1 frame)")
    # plt.xlabel("Time index")
    # plt.ylabel("Rotational Velocity (rad/s)")
    # plt.legend(ncol=3)
    # plt.grid(True)
    # plt.tight_layout()

    # plt.show()