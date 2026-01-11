import numpy as np
from scipy.spatial.transform import Rotation as R
from ahrs.filters import Madgwick
import matplotlib.pyplot as plt
import os 
import pandas as pd
import torch
import torch.nn as nn
import time
import sys

def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        # Write to stdout, overwriting the previous line
        sys.stdout.write('\r' + timer + ' ')
        sys.stdout.flush()
        time.sleep(1)
        t -= 1
    sys.stdout.write('\rComplete!   \n')

# Get input from the user for the total seconds
np.random.seed(0)
g = 9.80665
g_world = np.array([0.0, 0.0, g])  # gravity pointing +Z down in world

def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        # Write to stdout, overwriting the previous line
        sys.stdout.write('\r' + timer + ' ')
        sys.stdout.flush()
        time.sleep(1)
        t -= 1
    sys.stdout.write('\rComplete!   \n')


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


class SmallWav2Vec(nn.Module):
    def __init__(self, tokenizer, vocab_size=6):
        super().__init__()

        # ===== Feature extractor =====
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(24, 64, kernel_size=11, stride=3, padding=5),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4),
        ])

        self.act = nn.ReLU()

        # ===== Feature projection =====
        self.proj = nn.Linear(256, 128)

        # ===== Tiny transformer =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ===== CTC head =====
        self.ctc_head = nn.Linear(128, vocab_size)
        self.tokenizer = tokenizer


    def forward(self, x): #, seq_lengths=None, labels=None):
        """
        x: [B, 1, T]
        seq_lengths: original valid lengths BEFORE padding (tensor [B])
        """

        # =============================
        # 1. Pass through convs
        # =============================
        for conv in self.conv_layers:
            x = self.act(conv(x))            # [B, C, T']

        # =============================
        # 2. Feature projection
        # =============================
        x = x.transpose(1, 2)               # [B, T', 256]
        x = self.proj(x)                    # [B, T', 128]

        # =============================
        # 3. Build transformer mask
        # =============================
        # transformer expects mask: True = BLOCK
        max_len = x.size(1)
        mask = torch.zeros((max_len,)).to(x.device).unsqueeze(0)
        # mask: [B, T'] boolean

        # =============================
        # 4. Transformer encoder
        # =============================
        x = self.transformer(x, src_key_padding_mask=mask)   # [B, T', 128]

        # =============================
        # 5. LM head → logits
        # =============================
        logits = self.ctc_head(x)  

        return type('Output', (object,), {'logits': logits})()



class CharTokenizer:
    def __init__(self, characters):
        # characters: string like " abcdefghijklmnopqrstuvwxyz'"
        self.chars = characters
        self.char2id = {c: i for i, c in enumerate(characters)}
        self.id2char = {i: c for c, i in self.char2id.items()}
        self.blank_id = len(self.chars)  # last one for CTC

    def encode(self, text: str):
        return [self.char2id[c] for c in text if c in self.char2id]

    def decode(self, ids):
        return "".join([self.id2char[i] for i in ids if i in self.id2char])



def ctc_decode(logits, tokenizer: CharTokenizer):
    """
    logits: [B, L, vocab_size]
    returns: list of decoded strings
    """
    pred_ids = torch.argmax(logits, dim=-1)  # [B,L]
    decoded = []
    for ids in pred_ids:
        collapsed = []
        prev = None
        for i in ids:
            i = i.item()
            if i != tokenizer.blank_id and i != prev:  # treat blank_id as CTC blank
                collapsed.append(i)
            prev = i
        decoded.append(tokenizer.decode(collapsed))
    return decoded


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: ', device)

tokenizer = CharTokenizer('beign')

model = SmallWav2Vec(tokenizer, vocab_size=6)
model.load_state_dict(torch.load("FINAL_DIFF6.pth"))
model = model.to(device)
model.eval()


last_read = 0
step_size = 200

word_so_far = ''

## assumes hand at rest briefly at every restart time, otherwise Madgwick filter doesn't work 

while True: 
    csv_file = "TRIALS/3.csv"
    df = pd.read_csv(csv_file)
    if not len(df) > last_read + step_size: # rerun model each 2 seconds or step_size whatever longer 
        time.sleep(0.1)
        continue
    
    # Create a dictionary: key = sensor_index, value = DataFrame for that sensor
    dfs = {sensor: group for sensor, group in df[last_read:].groupby("sensor_index")}

    # if last_read == 0: # ONLY SCALE CORRECT THE FIRST TIME, reuse the rest of the time
    scale_correction = np.array([1.0 for _ in range(len(dfs))])

    for d in range(len(dfs)): 
        x = dfs[d].loc[:, 'ax':'az'].iloc[0:5] # average first 5 measurements
        scale_correction[d] = 9.80665/np.mean(np.linalg.norm(x, axis=1))

    # print('Scale correction: ', scale_correction)

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
        t = delta_t

        # Run Madgwick filter using accelerometer (should be reliable because motion is slow)
        madgwick = Madgwick(beta=0.0001)  # small-mid beta: allow accel to correct
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

    df_clean = pd.DataFrame(data, columns=columns)
    #### CLEANED DATA FOR MODEL
    sequence = df_clean[['acc1_x','acc1_y','acc1_z','gyro1_x','gyro1_y','gyro1_z','acc2_x','acc2_y','acc2_z','gyro2_x','gyro2_y','gyro2_z', 'acc3_x','acc3_y','acc3_z','gyro3_x','gyro3_y','gyro3_z','acc4_x','acc4_y','acc4_z','gyro4_x','gyro4_y', 'gyro4_z']].to_numpy()

    waveform = torch.tensor(sequence)

    # per-channel normalization and scale to [-1, 1]
    waveform = (waveform - waveform.mean(dim=0)) / (waveform.std(dim=0) + 1e-5)
    waveform = waveform / waveform.abs().max()

    sl = waveform.shape[-1]
    waves = waveform.permute(1,0).float().to(device).unsqueeze(0)

    last_read = len(df)

    with torch.no_grad():

        logits = model(waves).logits # [B,L,V]
        pred_texts = ctc_decode(logits, tokenizer)[0]

        word_so_far += ' ' + pred_texts 
        # print(pred_texts)
        print(f"\rPredicted signs: {word_so_far}") #, end="")
        print('--------')


    countdown(8)
    print('--------')

    # time.sleep(5)

