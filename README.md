# Gesture Recognition in Termux

A tutorial of running a simple gesture recognition in Termux. You can check [this Reddit post](https://www.reddit.com/r/termux/comments/1q5jxev/smooth_computer_vision_on_android_python/) to see the actual result.

## Important Notes

1. **This project is for masochism only.** If you are looking for a production-ready solution, please refer to [MediaPipe Solutions](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer) or [Google ML Kit](https://developers.google.com/ml-kit) for native Android/iOS integration.
2. Everything below is guaranteed to work only on my specific device (OnePlus 13 24+1T, Android 15, ColorOS 15, Snapdragon 8 Elite). Since implementation details vary significantly across different hardware, expect it to fail on your device.
3. Your device should have at least 5 GB of free space.

## Overview

This project implements simple gesture recognition within Termux using the following components:
1. **[Termux](https://github.com/termux/termux-app)**: The foundation of this project.
2. **[FadCam](https://github.com/anonfaded/FadCam)**: Or any camera app capable of starting a camera server.
3. **[Termux:X11](https://github.com/termux/termux-x11)**: Used to display the final result.
4. **[MediaPipe](https://github.com/google-ai-edge/mediapipe)**: Used to extract hand landmarks.
5. **`model.pth`**: A lightweight pre-trained model included in this repository.
6. **[Termux Sandbox](https://github.com/788009/termux-sandbox)**: Used to create an Ubuntu environment via **chroot**.
    - **Note**: Root access is required for optimal speed. Alternatively, you can use [proot-distro](https://github.com/termux/proot-distro), though it comes with **significant performance overhead**.

## Preparation

### Prepare the camera server

> [!NOTE]
>
> Here I take FadCam as an example. Other choices of camera server should have similar setup steps.

1. Install FadCam and open it.
2. Tap the button at the bottom center.
3. Start the server.
4. Return to the home page and start the camera.
5. Open http://127.0.0.1:8080/ in your browser and verify the connection.
6. Once verified, stop the server and the camera to prevent your device from overheating.

### Install Termux

Install Termux from [GitHub releases](https://github.com/termux/termux-app/releases) or [F-Droid](https://f-droid.org/en/packages/com.termux/), but **not** from Google Play.

### Prepare Termux:X11

1. Install Termux:X11 from the same source as your Termux app.
2. Inside Termux, run the following scripts.
```bash
pkg update
pkg install x11-repo -y
```

### Create a sandbox

Following the guidance [here](https://github.com/788009/termux-sandbox/blob/main/ubuntu/README.md):

1. Install `ubuntu-sandbox`
2. Create a sandbox

### Setup the environment

Enter the sandbox.

> [!NOTE]
>
> Wi-Fi connection is recommended, and it may take a long time.

**Inside the sandbox**, run the following scripts:

```bash
apt update && apt upgrade -y
apt install curl python3-pip python3-venv libgl1 libglib2.0-0 xfce4 dbus-x11 -y
# You may have to choose your area

curl -L https://github.com/788009/gesture-recognition-in-termux/releases/download/latest/gesture.tar -o ~/gesture.tar
tar -xvf ~/gesture.tar -C ~
cd ~/gesture
```

Make sure you have entered `~/gesture`, and then:

```bash
python3 -m venv .env
source .env/bin/activate
pip install opencv-python mediapipe numpy pyyaml torch
```
> [!WARNING]
> 
> If you are using a camera server other than FadCam, ensure you update the stream URL in `config.yaml` accordingly.

## Run the script

1. Start the camera server.
2. Start X11 service **in the Termux host** (not inside the sandbox):
```bash
termux-x11 :0 -ac
```
3. Enter the sandbox **with `-b`** (unrestricted mode)
4. Enter Xfce4 desktop **inside the sandbox**:
```bash
mkdir -p /tmp/.X11-unix
mount --bind /host_root/data/data/com.termux/files/usr/tmp/.X11-unix /tmp/.X11-unix
export DISPLAY=:0
xfce4-session
```
5. Run the script **inside the Xfce4 desktop** (using the Terminal Emulator):
```bash
cd ~/gesture
source .env/bin/activate
python3 detect.py
```
6. Maximize the window.
7. Move your hand into the camera's field of view to see the result.

## Stop and exit

1. Stop the script: Press `Ctrl` + `C` in the Terminal Emulator.
2. Exit the desktop: Click the top-right button on your screen and log out.
3. Exit the sandbox: Press `Ctrl` + `D` in the session where you enter the sandbox.
4. Stop X11 service: Press `Ctrl` + `C` in the session where you start it.
5. Stop the camera server: Open FadCam and stop the server as well as the camera.

> [!WARNING]
>
> Simply swiping FadCam app away from your recent tasks will **not** terminate the server.

## Community

1. **Discussions**: Use this space to share your results or ask questions regarding specific phone models and operating systems.
2. **Issues**: If you find a logic error in the code or a flaw in the instructions, please open an issue with your device details and error logs.

## FAQ

1. **Q: Why is the frame rate (FPS) so low?**  
   **A**: Your processor lacks the necessary performance for this task. Hand landmark extraction and neural network inference are computationally intensive.
2. **Q: Why is the video smooth but the latency is so high?**  
   **A**: This is typically caused by the buffering mechanism of the network stream. Currently, there is no known solution to reduce the latency. You are encouraged to explore alternative camera apps capable of starting a camera server for potentially better results.
3. **Q: Why is the recognition inaccurate sometimes?**  
   **A**: The MLP model used here was developed as a school assignment and was not optimized for high precision.

## Support

If you like this project, please give it a star!

## License

MIT License