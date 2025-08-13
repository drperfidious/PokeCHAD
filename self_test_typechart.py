from Data.poke_env_moves_info import MovesInfo

EXPECTED = [
    ("Dark","Fairy",0.5),
    ("Fairy","Dark",2.0),
    ("Fire","Grass",2.0),
    ("Fire","Water",0.5),
    ("Water","Fire",2.0),
    ("Fighting","Ghost",0.0),
    ("Ghost","Normal",0.0),
    ("Fire","Ground",1.0),
    ("Fire","Poison",1.0),
]


def run_self_test(gen: int = 9) -> int:
    mi = MovesInfo(gen)
    tc = mi.get_type_chart()
    fails = []
    for atk, dfd, exp in EXPECTED:
        got = tc.get(atk, {}).get(dfd)
        if got is None:
            fails.append(f"MISSING {atk}->{dfd} (expected {exp})")
        elif abs(got - exp) > 1e-9:
            fails.append(f"WRONG {atk}->{dfd}: got {got} expected {exp}")
    # Clodsire neutrality check (Poison/Ground vs Fire)
    fire_vs_clodsire = tc.get('Fire', {}).get('Poison', 1.0) * tc.get('Fire', {}).get('Ground', 1.0)
    if abs(fire_vs_clodsire - 1.0) > 1e-9:
        fails.append(f"WRONG Fire vs Clodsire (Poison/Ground) composite: {fire_vs_clodsire} expected 1.0")

    print("Type Chart Self-Test")
    print("====================")
    for atk, dfd, exp in EXPECTED:
        print(f"{atk:9s} -> {dfd:7s} = {tc.get(atk, {}).get(dfd)} (expected {exp})")
    print(f"Fire vs Clodsire (Poison/Ground) composite = {fire_vs_clodsire} (expected 1.0)")

    if fails:
        print("\nFAILURES:")
        for f in fails:
            print(" -", f)
        return 1
    print("\nAll checks passed.")
    return 0

if __name__ == "__main__":
    code = run_self_test()
    raise SystemExit(code)

