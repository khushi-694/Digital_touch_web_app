from flask import Flask, render_template_string, request, jsonify, send_file
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import serial
from datetime import datetime

app = Flask(__name__)

# Configuration
SERIAL_PORT = 'COM23'  # Update to your Arduino COM port
BAUD_RATE = 9600
SOFT_HARD_THRESHOLD = 350
FRESH_ROTTEN_THRESHOLD = 750

# Global variables
state = "Idle"
stop_requested = False
classification_type = "soft_hard"
current_test_config = {
    'cycles': 3,
    'duration': 5,
    'threshold': SOFT_HARD_THRESHOLD
}
test_data = {
    'all_data': [],
    'untouch_data': [],  # New: To store untouch data separately
    'touch_data': [],    # To store touch data separately
    'average_peak_value': 0,
    'touch_max_array': [],
    'labels': [],
    'finished': False
}
test_start_time = None

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Sensor Test Interface | Arduino Nano 33 IoT</title>
  <script src="https://kit.fontawesome.com/a076d05399.js" crossorigin="anonymous"></script>
  <style>
    :root {
      --bg-light: #f9f9f9;
      --bg-dark: #121212;
      --text-light: #002B5B;
      --text-dark: #f0f0f0;
      --card-bg-light: #fff;
      --card-bg-dark: #1e1e1e;
    }

    body {
      margin: 0;
      padding: 0;
      font-family: 'Segoe UI', sans-serif;
      background-color: var(--bg-light);
      color: var(--text-light);
      transition: background-color 0.3s ease, color 0.3s ease;
    }

    body.dark {
      background-color: var(--bg-dark);
      color: var(--text-dark);
    }

    .container {
      max-width: 860px;
      margin: 40px auto;
      background: var(--card-bg-light);
      border-radius: 12px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
      padding: 40px;
      transition: background-color 0.3s ease;
    }

    body.dark .container {
      background: var(--card-bg-dark);
    }

    h2 {
      text-align: center;
      margin-bottom: 35px;
    }

    label {
      font-weight: 600;
      display: block;
      margin-bottom: 8px;
    }

    input[type="number"], select {
      width: 100%;
      padding: 12px;
      font-size: 1em;
      border: 1px solid #ccc;
      border-radius: 8px;
      margin-bottom: 20px;
      background-color: inherit;
      color: inherit;
    }

    .radio-group {
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      margin-bottom: 20px;
    }

    .radio-option {
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 5px 0;
    }

    button, a.button {
      padding: 12px 18px;
      font-size: 1em;
      font-weight: 600;
      color: #fff;
      background-color: #0077cc;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.3s ease, transform 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }

    button:hover, a.button:hover {
      background-color: #005fa3;
      transform: scale(1.02);
    }

    button:disabled {
      background-color: #888;
      cursor: not-allowed;
    }

    .info-box {
      padding: 14px 18px;
      margin-top: 20px;
      border-radius: 6px;
      font-size: 1em;
    }

    #status {
      background-color: #f0f8ff;
      border-left: 6px solid #17a2b8;
    }

    #status.running {
      border-left-color: #ffc107;
      background-color: #fff8e1;
    }

    #status.success {
      border-left-color: #28a745;
      background-color: #e8f5e9;
    }

    #status.error {
      border-left-color: #dc3545;
      background-color: #f8d7da;
    }

    #average, #result {
      background-color: #eaf6ff;
      border-left: 5px solid #007bff;
      font-size: 1.1em;
      font-weight: 600;
    }

    #timer {
        background-color: #f0f8ff;
        border-left: 6px solid #17a2b8;
        margin-top: 10px;
    }

    #plotArea {
      margin-top: 30px;
    }

    img#plotImg {
      border: 1px solid #ccc;
      border-radius: 10px;
      max-width: 100%;
      margin-bottom: 15px;
    }

    .download-buttons {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    a.button {
      background-color: #28a745;
      text-decoration: none;
    }

    a.button:hover {
      background-color: #218838;
    }

    .hidden {
      display: none;
    }

    .loading {
      display: inline-block;
      width: 18px;
      height: 18px;
      border: 3px solid #f3f3f3;
      border-top: 3px solid #007bff;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    
    .top-bar {
      display: flex;
      justify-content: flex-end;
      margin-bottom: 10px;
    }

    .dark-toggle {
      margin-right: 10px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .dark-toggle input {
      transform: scale(1.2);
    }

    @media (max-width: 600px) {
      .container {
        padding: 25px;
      }

      .radio-group {
        flex-direction: column;
      }

      button {
        width: 100%;
        margin-bottom: 10px;
      }
    }
    body.dark #status,
    body.dark #average,
    body.dark #result,
    body.dark #timer {
      background-color: #1c2938;
      color: #d6e9ff;
      border-left-color: #4dabf7;
    }

    body.dark #status.success {
      background-color: #1e3521;
      border-left-color: #4caf50;
      color: #d1f0d2;
    }

    body.dark #status.error {
      background-color: #3b1d1d;
      border-left-color: #f44336;
      color: #f4cccc;
    }

    body.dark #status.running {
      background-color: #403914;
      border-left-color: #ffc107;
      color: #fff1a3;
    }
    body.dark a.button {
      background-color: #388e3c;
    }

    body.dark a.button:hover {
      background-color: #2e7d32;
    }
  </style>
</head>
<body>
<div class="container">
  <div class="top-bar">
    <div class="dark-toggle">
      <label for="darkSwitch"><i class="fas fa-moon"></i> Dark Mode</label>
      <input type="checkbox" id="darkSwitch">
    </div>
  </div>

  <h2>Arduino Nano 33 IoT Sensor Test</h2>

  <form id="testForm">
    <label>Classification Type:</label>
    <div class="radio-group">
      <div class="radio-option">
        <input type="radio" id="soft_hard" name="classification_type" value="soft_hard" checked>
        <label for="soft_hard">Soft / Hard</label>
      </div>
      <div class="radio-option">
        <input type="radio" id="fresh_rotten" name="classification_type" value="fresh_rotten">
        <label for="fresh_rotten">Fresh / Rotten</label>
      </div>
    </div>

    <label for="cycles">Number of Cycles</label>
    <input type="number" id="cycles" value="3" min="1">
    <div id="softHardThresholdGroup">
      <label for="softThreshold">Soft/Hard Threshold</label>
      <input type="number" id="softThreshold" value="350" min="0">
    </div>

    <div id="freshRottenThresholdGroup" class="hidden">
      <label for="freshThreshold">Fresh/Rotten Threshold</label>
      <input type="number" id="freshThreshold" value="750" min="0">
    </div>

      <label for="duration">Duration per Cycle (seconds)</label>
      <input type="number" id="duration" value="5" min="1">

    <button type="submit" id="startBtn">
      <i class="fas fa-play"></i> Start Test
    </button>

    <button type="button" id="stopBtn">
      <i class="fas fa-stop"></i> Stop Test
    </button>
  </form>

  <div id="status" class="info-box">Status: {{ status }}</div>
  <div id="timer" class="info-box hidden">Elapsed Time: 0s</div>
  <div id="average" class="info-box hidden"></div>
  <div id="result" class="info-box hidden"></div>

  <div id="plotArea" class="hidden">
    <h3>Sensor Data Plot</h3>
    <img id="plotImg" src="/plot?t={{ timestamp }}" alt="Sensor Plot">
    <br>
    <div class="download-buttons">
      <a href="/download_all" class="button"><i class="fas fa-download"></i> Download All Data CSV</a>
      <a href="/download_touch" class="button"><i class="fas fa-download"></i> Download Touch Data CSV</a>
      <a href="/download_untouch" class="button"><i class="fas fa-download"></i> Download Untouch Data CSV</a>
    </div>
  </div>
</div>

<script>
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const statusBox = document.getElementById("status");
  const timerBox = document.getElementById("timer");

  // Variables for timer offset
  let timerDisplayOffset = 0;
  let timerStartedDisplaying = false; // Flag to track when timer first becomes visible

  // Dark mode
  const darkSwitch = document.getElementById('darkSwitch');
  if (localStorage.getItem("dark-mode") === "true") {
    document.body.classList.add('dark');
    darkSwitch.checked = true;
  }

  darkSwitch.addEventListener('change', () => {
    document.body.classList.toggle('dark');
    localStorage.setItem("dark-mode", darkSwitch.checked);
  });

  const softGroup = document.getElementById("softHardThresholdGroup");
  const freshGroup = document.getElementById("freshRottenThresholdGroup");
  document.querySelectorAll('input[name="classification_type"]').forEach(radio => {
    radio.addEventListener("change", () => {
      if (radio.checked && radio.value === "soft_hard") {
        softGroup.classList.remove("hidden");
        freshGroup.classList.add("hidden");
      } else {
        softGroup.classList.add("hidden");
        freshGroup.classList.remove("hidden");
      }
    });
  });


  document.getElementById("testForm").onsubmit = async function(e) {
    e.preventDefault();
    const classificationType = document.querySelector('input[name="classification_type"]:checked').value;
    const cycles = document.getElementById("cycles").value;
    const duration = document.getElementById("duration").value;
    const softThreshold = document.getElementById("softThreshold").value;
    const freshThreshold = document.getElementById("freshThreshold").value;

    // Reset UI and timer state for a new test
    statusBox.className = 'info-box running';
    statusBox.innerText = "Status: Starting test..."; // Initial status during the delay
    timerBox.classList.add("hidden"); // Hide timer initially
    timerStartedDisplaying = false; // Reset the flag
    timerDisplayOffset = 0; // Reset the offset
    document.getElementById("result").classList.add("hidden");
    document.getElementById("average").classList.add("hidden");
    document.getElementById("plotArea").classList.add("hidden");

    // Spinner ON (start button disabled)
    startBtn.disabled = true;

    const res = await fetch('/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        classification_type: classificationType,
        cycles: cycles,
        duration: duration,
        soft_threshold: softThreshold,
        fresh_threshold: freshThreshold
      })
    });

    const data = await res.json();
    startBtn.disabled = false;
    // Status text will be updated by updateStatus polling
  };


  stopBtn.onclick = async function() {
    const confirmStop = confirm("Are you sure you want to stop the test?");
    if (!confirmStop) return;

    await fetch('/stop');
    statusBox.className = 'info-box error';
    statusBox.innerText = "Status: Test stopped by user.";
    timerBox.classList.add("hidden"); // Hide timer on stop
    timerBox.innerText = "Elapsed Time: 0s";
    timerStartedDisplaying = false; // Reset flag
    timerDisplayOffset = 0; // Reset offset
  };

  async function updateStatus() {
    try {
      const res = await fetch('/status');
      const data = await res.json();

      if (data.status) {
        statusBox.innerText = "Status: " + data.status;
      }

      // **Modified timer update logic**
      if (data.status.includes("Cycle")) {
        if (!timerStartedDisplaying) {
            // This is the first time we're seeing a "Cycle" status
            // data.elapsed_time will be the time passed since test_start_time was set in backend
            // We set this as our offset so subsequent counts start from 0
            timerDisplayOffset = data.elapsed_time;
            timerStartedDisplaying = true;
            timerBox.classList.remove("hidden");
            timerBox.innerText = "Elapsed Time: 0s"; // Display 0 at the start of measurement
        } else {
            // For subsequent updates, subtract the initial offset
            // Ensure the time doesn't go negative due to minor sync issues
            let displayTime = Math.max(0, data.elapsed_time - timerDisplayOffset);
            timerBox.innerText = "Elapsed Time: " + displayTime + "s";
        }
      } else if (data.finished) {
            timerBox.classList.remove("hidden");
            // For total time, display the actual elapsed time from backend (no offset needed for final display)
            timerBox.innerText = "Total Time: " + data.elapsed_time + "s";
            // Reset flags for next test cycle
            timerStartedDisplaying = false;
            timerDisplayOffset = 0;
      } else { // Idle, starting, or stopped before a cycle began
            timerBox.classList.add("hidden");
            timerBox.innerText = "Elapsed Time: 0s"; // Reset visual for next run
            timerStartedDisplaying = false; // Reset flag
            timerDisplayOffset = 0; // Reset offset
      }

      if (data.finished) {
        statusBox.className = 'info-box success';
        const currentType = document.querySelector('input[name="classification_type"]:checked').value;

        if (currentType === "soft_hard" && data.average !== null && data.average !== undefined) {
          document.getElementById("average").innerText = "Average of Touch Peaks: " + data.average.toFixed(2);
          document.getElementById("average").classList.remove("hidden");
        } else {
          document.getElementById("average").classList.add("hidden");
        }

        document.getElementById("result").innerText = "Classification: " + data.result;
        document.getElementById("result").classList.remove("hidden");

        document.getElementById("plotImg").src = "/plot?t=" + new Date().getTime();
        document.getElementById("plotArea").classList.remove("hidden");
      }
    } catch (error) {
      console.error("Error updating status:", error);
    }

    setTimeout(updateStatus, 1000);
  }

  updateStatus();
</script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, status=state, timestamp=datetime.now().timestamp())

@app.route('/start', methods=['POST'])
def start():
    global current_test_config, stop_requested, state, test_data, classification_type, test_start_time

    content = request.get_json()
    classification_type = content['classification_type']
    soft_threshold = int(content.get('soft_threshold', SOFT_HARD_THRESHOLD))
    fresh_threshold = int(content.get('fresh_threshold', FRESH_ROTTEN_THRESHOLD))
    current_test_config = {
        'cycles': int(content['cycles']),
        'duration': int(content['duration']),
        'threshold': soft_threshold if classification_type == 'soft_hard' else fresh_threshold
    }

    test_data = {
        'all_data': [],
        'untouch_data': [],
        'touch_data': [],
        'average_peak_value': 0,
        'touch_max_array': [],
        'labels': [],
        'finished': False
    }

    stop_requested = False
    state = "Starting test..."
    test_start_time = time.time()
    thread = threading.Thread(target=run_test)
    thread.start()

    return jsonify({"message": "Test started..."})

@app.route('/stop')
def stop():
    global stop_requested, state
    stop_requested = True
    state = "Test stopped by user"
    return jsonify({"message": "Stopping..."})

@app.route('/status')
def get_status():
    global test_start_time
    elapsed_time = 0
    if test_start_time and not test_data['finished']:
        elapsed_time = time.time() - test_start_time
    elif test_data['finished'] and test_start_time:
        elapsed_time = time.time() - test_start_time
        test_start_time = None # Reset test_start_time after test completes

    return jsonify({
        "status": state,
        "finished": test_data['finished'],
        "result": test_data['labels'][-1] if test_data['labels'] else "No result yet",
        "average": test_data['average_peak_value'] if classification_type == 'soft_hard' else None,
        "classification_type": classification_type,
        "elapsed_time": int(elapsed_time)
    })

def run_test():
    global state, test_data
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        state = f"Serial Error: {e}"
        test_data['finished'] = True
        return

    try:
        for i in range(current_test_config['cycles']):
            if stop_requested:
                state = "Test Stopped"
                break
            
            # Collect UNTOUCH data
            state = f"Cycle {i+1}/{current_test_config['cycles']}: UNTOUCH"
            untouch_segment = collect_data(ser, current_test_config['duration'])
            test_data['all_data'].extend(untouch_segment)
            test_data['untouch_data'].extend(untouch_segment) # Store untouch data separately
            
            if stop_requested:
                state = "Test Stopped"
                break

            # Collect TOUCH data
            state = f"Cycle {i+1}/{current_test_config['cycles']}: TOUCH"
            touch_segment = collect_data(ser, current_test_config['duration'])
            test_data['all_data'].extend(touch_segment)
            test_data['touch_data'].extend(touch_segment) # Store touch data separately

        if not stop_requested:
            if classification_type == 'soft_hard':
                process_soft_hard_classification()
            else:
                process_fresh_rotten_classification()

            save_csv()
            plot_all()
            state = "Test Complete"
    finally:
        if not test_data['labels'] and not stop_requested: # Add a default label if no classification occurred and not stopped by user
             test_data['labels'].append("No Classification (Test Interrupted or Error)")
        elif stop_requested and not test_data['labels']:
             test_data['labels'].append("Test Stopped by User")
        
        test_data['finished'] = True
        try:
            ser.close()
        except:
            pass # Ignore errors if serial port is already closed or not open

def collect_data(ser, duration):
    segment = []
    start_time = time.time()
    while time.time() - start_time < duration:
        if stop_requested:
            break
        try:
            line = ser.readline().decode('ascii', errors='ignore').strip()
            if not line:
                continue
            values = [int(v.strip()) for v in line.split(',') if v.strip().isdigit()]
            if len(values) != 9:
                continue
            segment.append(values)
        except Exception:
            continue
    return segment

def process_soft_hard_classification():
    try:
        if not test_data['touch_data']:
            test_data['labels'].append("No Touch Data Collected")
            return

        df_touch = pd.DataFrame(test_data['touch_data'], columns=["Time", "TX", "RX1", "RX2", "RX3", "RX4", "RX5", "RX6", "RX7"])
        rx_only = df_touch[["RX1","RX2","RX3", "RX4", "RX5", "RX6", "RX7"]]
        max_vals_across_rx = rx_only.values.max()

        # Append to touch_max_array (if you want to track all maxes)
        test_data['touch_max_array'].append(max_vals_across_rx)

        # Calculate the average of all recorded touch maximums
        if test_data['touch_max_array']:
            test_data['average_peak_value'] = np.mean(test_data['touch_max_array'])
        else:
            test_data['average_peak_value'] = 0 # Or handle as appropriate if no peaks collected

        is_hard = max_vals_across_rx > current_test_config['threshold'] # Your original logic
        label = "Hard" if is_hard else "Soft"
        test_data['labels'].append(label)
    except Exception as e:
        state = f"Processing error (Soft/Hard): {str(e)}"
        test_data['labels'].append("Error")


def process_fresh_rotten_classification():
    try:
        # Use test_data['touch_data'] directly for classification
        if not test_data['touch_data']:
            test_data['labels'].append("No Touch Data Collected")
            return

        df_touch = pd.DataFrame(test_data['touch_data'], columns=["Time", "TX", "RX1", "RX2", "RX3", "RX4", "RX5", "RX6", "RX7"])
        rx_only = df_touch[["RX1","RX2","RX3", "RX4", "RX5", "RX6", "RX7"]]
        
        # Find the maximum value across all RX sensors for the entire touch period
        max_vals_across_rx = rx_only.values.max() # Get the single highest value from all RX columns in touch data
        test_data['touch_max_array'].append(max_vals_across_rx) # Store this for potential future analysis

        is_fresh = max_vals_across_rx > current_test_config['threshold']
        label = "Fresh" if is_fresh else "Rotten"
        test_data['labels'].append(label)
    except Exception as e:
        state = f"Processing error (Fresh/Rotten): {str(e)}"
        test_data['labels'].append("Error")

def save_csv():
    try:
        # Save all data
        df_all = pd.DataFrame(test_data['all_data'], columns=["Time", "TX", "RX1", "RX2", "RX3", "RX4", "RX5", "RX6", "RX7"])
        if not df_all.empty:
            df_all["NewTime"] = df_all["Time"] - df_all["Time"].iloc[0]
            df_all.to_csv("all_data.csv", index=False)
        else:
            state = "No 'all_data' to save to CSV."
            # Create empty file to prevent 404 on download
            open("all_data.csv", 'a').close() 

        # Save untouch data
        df_untouch = pd.DataFrame(test_data['untouch_data'], columns=["Time", "TX", "RX1", "RX2", "RX3", "RX4", "RX5", "RX6", "RX7"])
        if not df_untouch.empty:
            df_untouch["NewTime"] = df_untouch["Time"] - df_untouch["Time"].iloc[0]
            df_untouch.to_csv("untouch_data.csv", index=False)
        else:
            state = "No 'untouch_data' to save to CSV."
            open("untouch_data.csv", 'a').close() 

        # Save touch data
        df_touch = pd.DataFrame(test_data['touch_data'], columns=["Time", "TX", "RX1", "RX2", "RX3", "RX4", "RX5", "RX6", "RX7"])
        if not df_touch.empty:
            df_touch["NewTime"] = df_touch["Time"] - df_touch["Time"].iloc[0]
            df_touch.to_csv("touch_data.csv", index=False)
        else:
            state = "No 'touch_data' to save to CSV."
            open("touch_data.csv", 'a').close() 

    except Exception as e:
        state = f"Error saving CSVs: {str(e)}"

def plot_all():
    try:
        if not test_data['all_data']:
            state = "No data to plot."
            return

        df = pd.DataFrame(test_data['all_data'], columns=["Time", "TX", "RX1", "RX2", "RX3", "RX4", "RX5", "RX6", "RX7"])
        df["NewTime"] = df["Time"] - df["Time"].iloc[0]
        x = df["NewTime"]
        plt.figure(figsize=(10, 6))
        for col in ["RX1", "RX2", "RX3", "RX4", "RX5", "RX6", "RX7"]:
            plt.plot(x, df[col], label=col)
        plt.xlabel("Time (ms)")
        plt.ylabel("Sensor Value")
        plt.title(f"Sensor Data ({'Soft/Hard' if classification_type == 'soft_hard' else 'Fresh/Rotten'})")
        plt.legend()
        plt.grid(True) # Add grid for better readability
        plt.tight_layout()
        plt.savefig("all_data_plot.png")
        plt.close()
    except Exception as e:
        state = f"Error generating plot: {str(e)}"

@app.route('/download_all')
def download_all_csv():
    file_path = os.path.abspath("all_data.csv")
    if not os.path.exists(file_path):
        return "All Data CSV not found", 404
    return send_file(file_path, as_attachment=True, download_name="all_sensor_data.csv")

@app.route('/download_touch')
def download_touch_csv():
    file_path = os.path.abspath("touch_data.csv")
    if not os.path.exists(file_path):
        return "Touch Data CSV not found", 404
    return send_file(file_path, as_attachment=True, download_name="touch_sensor_data.csv")

@app.route('/download_untouch')
def download_untouch_csv():
    file_path = os.path.abspath("untouch_data.csv")
    if not os.path.exists(file_path):
        return "Untouch Data CSV not found", 404
    return send_file(file_path, as_attachment=True, download_name="untouch_sensor_data.csv")


@app.route('/plot')
def plot_img():
    plot_path = os.path.abspath("all_data_plot.png")
    if not os.path.exists(plot_path):
        return "Plot not found", 404
    return send_file(plot_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5000)