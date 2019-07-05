from phue import Bridge
import logging
logging.basicConfig()
import math
import time

b = Bridge('192.168.86.25')

print b.get_api()

#b.set_light('Hue color light 1', 'hue', 65536*3/4)

brightness_frequency = 1 #Hz
hue_frequency = 0.25 #Hz
power_frequency = 1 #Hz
while True:
    brightness = int((math.sin(time.time() * brightness_frequency * 2*math.pi) + 1) / 2 * 254)
    hue = int((math.sin(time.time() * hue_frequency * 2*math.pi) + 1) / 2 * 65536)
    power = int(math.sin(time.time() * hue_frequency * 2*math.pi) + 1)
    brightness = 254
    power = True
    b.set_light('Hue color light 1', 'on', bool(power))
    if power:
        b.set_light('Hue color light 1', 'bri', brightness)
        b.set_light('Hue color light 1', 'hue', hue)
#    print brightness
