from state import *

state = create_state(
    "Generate a password and tell me the current time."
)

print("Initial State")
print(state)

add_action(
    state,
    "generate_password"
)

next_step(state)

record_observations(
    state,
    "generate_password",
    "Password : X#8Lm@4P"
)



print("\nState After One Iteration")
print(state)

finish(
    state,
     "Password : X#8Lm@4P\nCurrent Time : 10:35 AM"
)

print("\nFinal State")
print(state)
