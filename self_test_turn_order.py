# Simple self-test for turn order speed-first probability
from Data.turn_order import speed_first_probability

def run():
    cases = [
        (200, 300, False, 0.0),
        (300, 200, False, 1.0),
        (200, 200, False, 0.5),
        (200, 300, True, 1.0),
        (300, 200, True, 0.0),
    ]
    for us, os, tr, exp in cases:
        got = speed_first_probability(us, os, tr)
        print(f"speed_first_probability({us}, {os}, trick_room={tr}) = {got}")
        assert abs(got - exp) < 1e-9, f"Expected {exp}, got {got}"
    print("OK: speed_first_probability tests passed.")

if __name__ == "__main__":
    run()

