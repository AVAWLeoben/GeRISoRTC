# 🧠 SenSoRTC

**Se**nsor Based **So**rting **R**eal **T**ime **C**ontrol Software

Real-time object detection pipeline with **YOLO**, camera input (USB / Basler / simulated), and **nozzle activation control** (hardware or simulated).

Designed for conveyor-style inspection + actuation systems.

---

## 🚀 Features

* 🔍 **YOLO-based real-time detection**
* 📷 Multiple camera sources:

  * USB
  * Basler (Pylon)
  * Simulated video input
* 🎯 **Nozzle activation mask generation**
* ⚡ Multiprocessing pipeline (producer / consumer / UI)
* 🎮 Interactive **pygame UI**
* 🧪 Fully **simulatable system** (camera + nozzles)
* 📊 Optional **vertical movement calibration (tracking-based)**

---

## 🧱 Setup

### Basler Camera Pylon Software Suite

Download:
https://www.baslerweb.com/de-de/downloads/software/143940061/

Install the latest version (e.g. **Pylon 26.04** or newer).

---

### NVIDIA Jetson Thor (JetPack 7)

#### Using Prebuilt Environment
- Download Environments/SenSoRTC_THOR.yml
- Install the Environment via Anaconda
- Open Anaconda Prompt and cd into the download directory

- Then use Conda Env Create to recreate the THOR Environment
  
```bash
conda env create -f SenSoRTC_THOR.yml
```

#### Native Installation from Scratch
##### 1. Create Environment

```bash
conda create -n SenSoRTC python=3.12 spyder
conda activate SenSoRTC
```

---

##### 2. Install Ultralytics

```bash
sudo apt update
sudo apt install python3-pip -y
pip install -U pip

pip install ultralytics[export]
```

Reboot:

```bash
sudo reboot
```

---

##### 3. Install Jetson-Compatible PyTorch

⚠️ Default pip installs are **NOT compatible with Jetson (ARM64)**

The above ultralytics installation will install Torch and Torchvision. However, these 2 packages installed via pip are not compatible to run on Jetson AGX Thor which comes with JetPack 7.0 and CUDA 13. Therefore, we need to manually install them.

Install torch and torchvision according to JP7.0

Install:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

---


##### 4. Install ONNX Runtime (Jetson)
The onnxruntime-gpu package hosted in PyPI does not have aarch64 binaries for the Jetson. So we need to manually install this package. This package is needed for some of the exports.

Here we will download and install onnxruntime-gpu 1.24.0 with Python3.12 support.

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.24.0-cp312-cp312-linux_aarch64.whl
```

---

##### 5. Install Other Dependencies

```bash
pip install pygame-ce==2.5.6
pip install pymodbus==3.5.4
pip install pypylon==26.3.1
pip install pyserial==3.5
pip install easygui==0.98.3
pip install lap==0.5.13
```

---

---

### NVIDIA Jetson Orin/Nano (JetPack 6)

#### 1. Create Environment

```bash
conda create -n SenSoRTC python=3.10 spyder
conda activate SenSoRTC
```

---

#### 2. Install Ultralytics

```bash
sudo apt update
sudo apt install python3-pip -y
pip install -U pip

pip install ultralytics[export]
```

Reboot:

```bash
sudo reboot
```

---

#### 3. Install Jetson-Compatible PyTorch

⚠️ Default pip installs are **NOT compatible with Jetson (ARM64)**

Install:

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.10.0-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl
```

---

#### 4. Install cuDSS (required)

```bash
wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudss
```

---

#### 5. Install ONNX Runtime (Jetson)

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
```

---

#### 6. Install Other Dependencies

```bash
pip install pygame-ce==2.5.6
pip install pymodbus==3.5.4
pip install pypylon==26.3.1
pip install pyserial==3.5
pip install easygui==0.98.3
pip install lap==0.5.13
```

---

## 🧱 Architecture

```
Camera → YOLO → Mask → Queue → Nozzle Control
                 ↓
               Display UI
```

### Processes

| Process  | Role                                      |
| -------- | ----------------------------------------- |
| Producer | Camera + YOLO inference + mask generation |
| Consumer | Converts mask → nozzle activation         |
| Display  | UI + visualization + controls             |


## ▶️ Running the System

```bash
python SenSoRTC.py
```

Startup GUI allows configuration of:

* Camera type
* Connection type
* Model path
* Simulation input

---

## 🎛️ Configuration Options

### Camera Types

| Type      | Description               |
| --------- | ------------------------- |
| USB       | Standard webcam           |
| Basler    | Industrial camera (Pylon) |
| SIMULATED | Video file input          |

---

### Connection Types

| Type      | Description             |
| --------- | ----------------------- |
| UDP       | Binder HPNB |
| MODBUS    | EVK HPNB           |
| ARDUINO   | Serial control          |
| SIMULATED | Visual nozzle UI        |

---

## 🧪 Simulation Modes

### Simulated Camera

* Video file input
* Auto loop
* Resized to 640×640

### Simulated Nozzles

* Separate process
* 10 nozzle outputs
* Real-time visualization

---

## 🧠 Detection Pipeline

1. Frame acquisition
2. YOLO inference (`predict()` / `track()`)
3. Bounding box filtering
4. Mask generation

   * Vorschuss / Nachschuss / Beischuss
5. Temporal accumulation
6. Vertical shift compensation
7. Output to nozzle system

---

## 🎮 UI Controls

### General

| Key   | Action        |
| ----- | ------------- |
| ESC   | Quit          |
| SPACE | Record toggle |
| V / B | Vertical tune |

### Detection

| Key         | Action     |
| ----------- | ---------- |
| + / -       | Confidence |
| PgUp / PgDn | IoU        |
| Arrows      | Delay      |

### Mask Tuning

| Key   | Action     |
| ----- | ---------- |
| Q / A | Vorschuss  |
| W / S | Nachschuss |
| Y / X | Beischuss  |

---

## 📁 Important Files

| File                                | Purpose                    |
| ----------------------------------- | -------------------------- |
| AI_CLASSIFIER_CONTROL_SOFTWARE_*.py | Main pipeline              |
| NOZZLE_CONTROL_LAYER.py             | Hardware/simulated outputs |
| UI_LAYER.py                         | UI                         |
| STARTUP_SETTINGS_GUI.py             | Config GUI                 |
| simulated_camera.py                 | Video camera simulation    |

---

## ⚠️ Notes & Gotchas

* Multiprocessing uses `spawn`
* Pass all args explicitly (no globals)

**VIDEO_PATH bug:**

```python
connectCamera(..., VIDEO_PATH=VIDEO_PATH)
```

* Avoid image distortion (aspect ratio!)
* Remove any preprocessing that harms YOLO

**ARDUINO DRIVER MISSING:**
Reload Arduino CH341 Driver!

```bash
cd ~/ch341ser_linux/driver
sudo insmod ./ch341.ko
lsmod | grep ch34
dmesg | tail -20
ls /dev/ttyCH341* /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

---

## 🧩 Extending the System

### Add Nozzle Backend

```python
def nozzle_control_MY_BACKEND(queue, stop_flag):
    pass
```

---

### Add Camera

Extend:

```python
connectCamera(...)
```

---

## 🧪 Debug Tips

* Start with `SIMULATED + SIMULATED`
* Enable bounding boxes
* Print mask slices
* Monitor vertical calibration

---

## 📦 Output

* Camera frames → `Recordings_Camera/`
* UI frames → `Recordings_GUI/`

---

## 🏁 Summary

Modular **real-time vision + actuation system** with:

* multiprocessing pipeline
* pluggable IO layers
* full simulation support

Built for **industrial inspection systems** and rapid experimentation.

## USB Camera Controls on Jetson
✅ Install v4l2-ctl
```bash
sudo apt update
sudo apt install v4l2-utils -y
```

That gives you the v4l2-ctl command.

### Using the USB-Camera-Configurator
- Select USB Camera in Startup GUI
- Open USB_Camera_Configurator.
This opens a GUI using Video4Linux on Linux to search for USB Camera Controls.

### CLI Camera Controls
🔍 Then list camera controls

```bash
v4l2-ctl -d /dev/video0 --list-ctrls
```
Look specifically for:

focus_auto
focus_absolute
🎯 Example usage

#### Disable autofocus
v4l2-ctl -d /dev/video0 -c focus_auto=0

#### Set manual focus (example value)
v4l2-ctl -d /dev/video0 -c focus_absolute=30

⚠️ Important reality check

If those controls don’t appear in the list, your camera:

either has fixed focus
or doesn’t expose focus control via V4L2

In that case, OpenCV also won’t be able to control it.


### 💡 Approach Idea

For your SenSoRTC pipeline, the cleanest setup is:

##### 1. Set focus BEFORE running Python
v4l2-ctl -d /dev/video0 -c focus_auto=0
v4l2-ctl -d /dev/video0 -c focus_absolute=XX

##### 2. Then start your pipeline
python SenSoRTC.py

That way YOLO always gets stable, non-pulsing frames (autofocus may destroy detection stability).

💡 Live Approach
You can run v4l2-ctl while your Python/OpenCV program is running, and the camera will update focus live.

✅ Live tuning workflow
Start your pipeline:
python SenSoRTC.py
In another terminal, adjust focus in real time:
v4l2-ctl -d /dev/video0 -c focus_absolute=350
v4l2-ctl -d /dev/video0 -c focus_absolute=400
v4l2-ctl -d /dev/video0 -c focus_absolute=450
v4l2-ctl -d /dev/video0 -c focus_absolute=500

You’ll immediately see the effect in your UI.



