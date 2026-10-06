import string

from tools import (
    get_current_time,
    roll_dice,
    generate_password
)

def execute_tool(user_input):
    cleaned = user_input.lower().translate(str.maketrans("", "", string.punctuation))
    words = cleaned.split()


    if "time" in words or "clock" in words:
        return get_current_time()
    elif "dice" in words or "die" in words:
        return f"You rolled {roll_dice()}"
    elif "password" in words:
        return generate_password()
    else:
        return None