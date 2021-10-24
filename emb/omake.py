import sys
import tty
import termios
import unicodedata

def len_count(text):
    count = 0
    for c in text:
        if unicodedata.east_asian_width(c) in 'FWA':
            count += 2
        else:
            count += 1
    return count

def getkey():
    def getch():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setraw(fd)
        char =  sys.stdin.read(1)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return char

    CTRL_C = 3
    CTRL_Z = 26
    ESC = 27
    CR = 13
    BS = 127

    while True:
        key = ord(getch())
        if key in [CTRL_C, CTRL_Z]:
            return 'Stop'
        elif key == CR:
            return 'Enter'
        elif key == BS:
            return 'BS'
        elif key == ESC:
            key = ord(getch())
            if key == ord('['):
                key = ord(getch())
                if key == ord('A'):
                    return 'Up'
                elif key == ord('B'):
                    return 'Down'
                elif key == ord('C'):
                    return 'Right'
                elif key == ord('D'):
                    return 'Left'
        elif chr(key) in '!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_abcdefghijklmnopqrstuvwxyz{|}~ ':
            return chr(key)


def console_menu(title, choices, sub_text='', default_idx=0, min_width=0):
    def print_menu(title, choices, sub_text, min_width, selected_idx):
        width = max(min_width, len_count(title) + 6, len_count(sub_text) + 6, max(map(len_count, choices)) + 10)
        
        print('┏' + '━' * (width - 2) + '┓')
        print('┃' + ' ' * (width - 2) + '┃')
        print('┃' + title.center(width - len_count(title) + len(title) - 2) + '┃')
        print('┃' + ' ' * (width - 2) + '┃')
        print('┠' + '─' * (width - 2) + '┨')

        for idx, choice in enumerate(choices):
            if idx == selected_idx:
                color_set = '\033[47m\033[30m'
                color_reset = '\033[0m\033[0m'
            else:
                color_set = ''
                color_reset = ''

            print('┃' + ' ' * (width - 2) + '┃')
            print('┃ ' + color_set + '%2d.  ' % (idx + 1) + choice.ljust(width - len_count(choice) + len(choice) - 9) + color_reset + ' ┃')
            
        print('┃' + ' ' * (width - 2) + '┃')
        if len(sub_text) > 0:
            print('┠' + '─' * (width - 2) + '┨')
            print('┃' + sub_text.center(width - len_count(sub_text) + len(sub_text) - 2) + '┃')
            print('┗' + '━' * (width - 2) + '┛', end='')
        else:
            print('┗' + '━' * (width - 2) + '┛')
            print()
            print(end='')


    selected_idx = default_idx
    print_menu(title, choices, sub_text, min_width, selected_idx)
    while True:
        print(f'\033[{len(choices) * 2 + 9}A')
        print_menu(title, choices, sub_text, min_width, selected_idx)
        
        key = getkey()
        if key in ['Stop', 'q']:
            selected_idx = -1
            break
        elif key == 'Enter':
            break
        elif key == 'Up':
            selected_idx -= 1
            if selected_idx < 0:
                selected_idx = len(choices) - 1
        elif key == 'Down':
            selected_idx += 1
            if selected_idx >= len(choices):
                selected_idx = 0
        elif key == 'Left':
            selected_idx = 0
        elif key == 'Right':
            selected_idx = len(choices) - 1
        elif key.isnumeric():
            if 0 < int(key) <= len(choices):
                selected_idx = int(key) - 1
                break

    print('\033[2K\033[1A' * (len(choices) * 2 + 9))

    return selected_idx

def console_inputarea(title, sub_text='', min_width=40, default_text='', numeric_ok=True, lowercase_ok=True, uppercase_ok=True, sign_ok=True):
    def print_inputarea(title, text, sub_text, min_width):
        width = max(min_width, len_count(title) + 6, len_count(sub_text) + 6)
        
        color_set = '\033[47m\033[30m'
        color_reset = '\033[0m\033[0m'
        
        print('┏' + '━' * (width - 2) + '┓')
        print('┃' + ' ' * (width - 2) + '┃')
        print('┃' + title.center(width - len_count(title) + len(title) - 2) + '┃')
        print('┃' + ' ' * (width - 2) + '┃')
        print('┃ ' + color_set + ' ' * (width - 4) + color_reset + ' ┃')
        print('┃ ' + color_set + '  > ' + text.ljust(width - len_count(text) + len(text) - 9) + ' ' + color_reset + ' ┃')
        print('┃ ' + color_set + ' ' * (width - 4) + color_reset + ' ┃')
        print('┃' + ' ' * (width - 2) + '┃')
        if len(sub_text) > 0:
            print('┠' + '─' * (width - 2) + '┨')
            print('┃' + sub_text.center(width - len_count(sub_text) + len(sub_text) - 2) + '┃')
            print('┗' + '━' * (width - 2) + '┛', end='')
        else:
            print('┗' + '━' * (width - 2) + '┛')
            print()
            print(end='')
            
    text = default_text
    print_inputarea(title, text, sub_text, min_width)
    while True:
        print(f'\033[11A')
        print_inputarea(title, text, sub_text, min_width)
        
        key = getkey()
        if key == 'Stop':
            text = None
            break
        elif key == 'Enter':
            break
        elif key == 'BS':
            text = text[:-1]
        elif numeric_ok and key.isnumeric():
            text += key
        elif key.isalpha():
            if lowercase_ok and key.islower():
                text += key
            elif uppercase_ok and key.isupper():
                text += key
        elif sign_ok and len(key) == 1:
            text += key

    print('\033[2K\033[1A' * 11)

    return text

def test():
    choices = ['stretch', 'padding', 'variable']
    selected_idx = console_menu('Select deform_type', choices)
    if selected_idx < 0:
        print('(Canceled)')
        return
    print(selected_idx, choices[selected_idx])

def test2():
    text = console_inputarea('Input text', 'Return to end', numeric_ok=True, lowercase_ok=False, uppercase_ok=False, sign_ok=False)
    if text is None:
        return
    print(text)
