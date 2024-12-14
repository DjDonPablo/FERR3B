from gpiozero import OutputDevice
from time import sleep


# Define GPIO to LCD mapping using OutputDevice
LCD_RS = OutputDevice(26)  # Pin 25 for RS
LCD_E = OutputDevice(19)  # Pin 24 for Enable
LCD_D4 = OutputDevice(13)  # Pin 23 for D4
LCD_D5 = OutputDevice(6)  # Pin 18 for D5
LCD_D6 = OutputDevice(5)  # Pin 15 for D6
LCD_D7 = OutputDevice(11)  # Pin 14 for D7

# Define LCD parameters
LCD_WIDTH = 16  # Maximum characters per line
LCD_CHR = True  # Mode - Sending data
LCD_CMD = False  # Mode - Sending command
LCD_LINE_1 = 0x80  # LCD RAM address for the 1st line
LCD_LINE_2 = 0xC0  # LCD RAM address for the 2nd line

# Timing constants
E_PULSE = 0.0005
E_DELAY = 0.0005


def lcd_init():
    """Initialize the LCD display."""
    lcd_byte(0x33, LCD_CMD)  # Initialize
    lcd_byte(0x32, LCD_CMD)  # Initialize
    lcd_byte(0x28, LCD_CMD)  # 2 line 5x7 matrix
    lcd_byte(0x0C, LCD_CMD)  # Turn cursor off
    lcd_byte(0x06, LCD_CMD)  # Shift cursor right
    lcd_byte(0x01, LCD_CMD)  # Clear display
    sleep(E_DELAY)


def lcd_byte(bits, mode):
    """Send byte to data pins."""
    # Send mode (True for data, False for command)
    LCD_RS.value = mode

    # High bits
    LCD_D4.off()
    LCD_D5.off()
    LCD_D6.off()
    LCD_D7.off()
    
    if bits & 0x10:
        LCD_D4.on()
    if bits & 0x20:
        LCD_D5.on()
    if bits & 0x40:
        LCD_D6.on()
    if bits & 0x80:
        LCD_D7.on()

    # Toggle 'Enable' pin
    lcd_toggle_enable()

    # Low bits
    LCD_D4.off()
    LCD_D5.off()
    LCD_D6.off()
    LCD_D7.off()
    
    if bits & 0x01:
        LCD_D4.on()
    if bits & 0x02:
        LCD_D5.on()
    if bits & 0x04:
        LCD_D6.on()
    if bits & 0x08:
        LCD_D7.on()

    # Toggle 'Enable' pin
    lcd_toggle_enable()


def lcd_toggle_enable():
    """Toggle enable pin."""
    sleep(E_DELAY)
    LCD_E.on()
    sleep(E_PULSE)
    LCD_E.off()
    sleep(E_DELAY)


def lcd_string(message, line):
    """Send string to display."""
    message = message.ljust(LCD_WIDTH, " ")

    lcd_byte(line, LCD_CMD)

    for i in range(LCD_WIDTH):
        lcd_byte(ord(message[i]), LCD_CHR)
