# FERR3B : Facial Emotion Recognition on Raspberry 3B+

### Commands to stream video flux from Raspberry to computer through SSH :

On Raspberry :
```
libcamera-vid -t 0 --inline --codec h264 -o - | nc -lkv4 5000
```
On computer :
```
nc <raspberry_ip> 5000 | ffplay -i -
```
