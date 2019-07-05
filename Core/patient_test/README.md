# AlterEgo Patient Test

**1. View usb device name and update bottom of patient_test_serial.py accordingly:**
```
ls /dev/tty.usbserial-*
```
---

**2. Open the OpenBCI GUI and record data via Cyton serial dongle**

All samples for a class should be in a single file. Turn on the board's digital read and hold down the PROG button on the board when recording a sample.

---

**3. Move saved data files to patient_data directory and rename to <class_index>.txt (eg. “0.txt”, “1.txt”, “2.txt”, etc.)**

NOTE: classes are determined by alphabetical order of filenames in the directory. If more than 10 classes, use letters instead of numbers

---

**4. Upload data files to media lab server:**
```
scp -r patient_data fluid@18.85.59.242:ewadkins
```
NOTE: Make sure that all previous data files are either overwritten or deleted, or they’ll be included in training

---

**5. Train on patient data**
```
ssh fluid@18.85.59.242

cd ewadkins
ipython patient_train.py
```
NOTE: Allow the training to finish (epoch 300). Best model is saved automatically.

---

**6. Download trained model from media lab server:**
```
exit

scp fluid@18.85.59.242:ewadkins/patient_model.ckpt.* .
```
---

**7. Run the real-time test**
```
ipython patient_test_serial.py
ipython patient_test_serial_trigger.py
ipython patient_test_serial_silence.py
```
NOTE: It’s probably best to wait until the SPU stabilizes (see terminal output) to ensure the proper rate of data processing. An SPU that is too low will result in data loss. An SPU that is too high results in unnecessary lag in the display. The stable SPU depends on processor speed and how computationally expensive the data processing/testing is.

NOTE: If test_model==True in patient_test_serial.py, the data will be ran against the trained model. Otherwise, the data is only visualized and not tested (which reduces lag on the display).

---

**Disclaimer: I use python 2, and this code may not work with python 3.**
