import dweepy

THING = "69484f63"

def add_to_display(message):
    dweepy.dweet_for(THING, {'append': message})
    
def clear_display():
    dweepy.dweet_for(THING, {'action': 'clear'})
    
def send_tts(message):
    dweepy.dweet_for(THING, {'tts': message})
