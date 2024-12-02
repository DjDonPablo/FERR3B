# FERR3B : Facial Emotion Recognition on Raspberry 3B+

### Config

- Raspberry Pi 3B+ with Raspberry Pi OS (Legacy, 64-bit)
- 8MP IMX219 from ArduCam
- an alphanumeric LCD (16x2 characters)

If using the same camera, here are the commands to make camera discoverable :

```
sudo nano /boot/firmware/config.txt 
#Find the line: camera_auto_detect=1, update it to:
camera_auto_detect=0
#Find the line: [all], add the following item under it:
dtoverlay=imx219
#Save and reboot.
```

For other ArduCam camera : https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/8MP-IMX219/

### Commands to stream video flux from Raspberry to computer through SSH :

On Raspberry :
```
libcamera-vid -t 0 --inline --codec h264 -o - | nc -lkv4 5000
```
On computer :
```
nc <raspberry_ip> 5000 | ffplay -i -
```
