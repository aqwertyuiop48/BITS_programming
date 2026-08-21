"""Expand the Titanic sample CSVs to a bakeoff-ready size.

Existing rows are preserved so simple-demo docs remain valid.
New rows are generated deterministically from a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.common import ensure_dir

# ─── Fixed rows that match the original small samples ────────────────────────

TRAIN_SEED_ROWS = [
    (1, 0, 3, "John Kelly",    "male",   34, 0, 0, "A/5 21171",           7.25,     "",     "S"),
    (2, 1, 1, "Anna Harper",   "female", 29, 1, 0, "PC 17599",           71.2833, "C85",    "C"),
    (3, 1, 3, "Maria Silva",   "female", 22, 0, 0, "STON/O2 3101282",     7.925,   "",      "S"),
    (4, 1, 1, "James Carter",  "male",   48, 1, 0, "113803",             53.1,    "C123",   "S"),
    (5, 0, 3, "Patrick Doyle", "male",   35, 0, 0, "373450",              8.05,    "",      "S"),
    (6, 0, 3, "Liam Murphy",   "male",   "", 0, 0, "330877",              8.4583,  "",      "Q"),
    (7, 0, 1, "Charles Kent",  "male",   54, 0, 0, "17463",             51.8625,  "E46",   "S"),
    (8, 1, 3, "Emily Stone",   "female", 18, 0, 0, "349909",              7.75,    "",      "Q"),
    (9, 1, 2, "Sarah Ali",     "female", 27, 0, 2, "237736",             13.0,    "",      "S"),
    (10,0, 2, "Robert Nash",   "male",   41, 0, 0, "PP 9549",            26.0,    "",      "S"),
    (11,1, 1, "Grace Lee",     "female", 36, 0, 0, "PC 17601",          76.7292,  "D33",   "C"),
    (12,0, 3, "Tom Baker",     "male",   19, 0, 0, "349245",              7.8958,  "",      "S"),
]

INFER_SEED_ROWS = [
    (101, 3, "Noah Reed",    "male",   28, 0, 0, "330911",           7.8958,  "",    "S"),
    (102, 1, "Olivia Chen",  "female", 32, 1, 0, "PC 17758",        82.1708, "B28", "C"),
    (103, 2, "Ethan Brooks", "male",   "", 0, 0, "SC/AH 3085",      10.5,    "",    "S"),
    (104, 3, "Ava Patel",    "female", 16, 0, 1, "367230",          12.2875,  "",   "S"),
    (105, 1, "Henry Cole",   "male",   45, 0, 0, "113783",          35.5,   "C68",  "S"),
]

MALE_NAMES   = ["William","Thomas","Charles","George","Arthur","Edward","Frederick",
                "Albert","Walter","Leonard","Herbert","Harold","Frank","Ernest",
                "Alfred","Percy","Reginald","Stanley","Clifford","Douglas"]
FEMALE_NAMES = ["Mary","Alice","Florence","Elizabeth","Dorothy","Margaret","Helen",
                "Edith","Ethel","Emma","Annie","Clara","Lily","Violet","Nellie",
                "Elsie","Mabel","Agnes","Beatrice","Constance"]
LAST_NAMES   = ["Smith","Jones","Williams","Brown","Taylor","Davies","Evans","Wilson",
                "Johnson","Harris","Martin","Clarke","Robinson","Lewis","Walker",
                "Hall","Allen","Young","Wright","Scott","King","Turner","Roberts",
                "Carter","Mitchell","Phillips","Cooper","Ward","Morris","Wood"]

CABIN_PREFIXES = ["A","B","C","D","E","F","G"]
TICKET_PREFIXES= ["PC","CA","SC","SOTON","LINE","WE/P","PP","C.A."]


def _survival_prob(sex: str, pclass: int, age: float | str) -> float:
    rates = {("female",1): 0.97, ("female",2): 0.86, ("female",3): 0.50,
             ("male",  1): 0.35, ("male",  2): 0.17, ("male",  3): 0.14}
    p = rates.get((sex, pclass), 0.38)
    if isinstance(age, (int, float)) and age < 15:
        p = min(0.95, p + 0.18)
    return p


def _ticket(rng: np.random.Generator, pclass: int) -> str:
    if pclass == 1 and rng.random() < 0.4:
        return f"{rng.choice(TICKET_PREFIXES)} {rng.integers(10000,99999)}"
    return str(rng.integers(100000, 999999))


def _cabin(rng: np.random.Generator, pclass: int) -> str:
    if pclass == 3 or rng.random() < 0.65:
        return ""
    return f"{rng.choice(CABIN_PREFIXES)}{rng.integers(10,150)}"


def _generate_rows(
    n: int,
    start_id: int,
    rng: np.random.Generator,
    include_survived: bool,
) -> list[tuple]:
    rows = []
    for i in range(n):
        pid   = start_id + i
        pc    = int(rng.choice([1,2,3], p=[0.22, 0.28, 0.50]))
        sex   = str(rng.choice(["male","female"], p=[0.55, 0.45]))
        names = MALE_NAMES if sex == "male" else FEMALE_NAMES
        name  = f"{rng.choice(names)} {rng.choice(LAST_NAMES)}"
        raw_age = float(round(rng.normal(28, 13), 0))
        raw_age = max(1.0, min(76.0, raw_age))
        age   = raw_age if rng.random() > 0.14 else ""
        sibsp = int(rng.choice([0,1,2], p=[0.68, 0.24, 0.08]))
        parch = int(rng.choice([0,1,2], p=[0.72, 0.20, 0.08]))
        base  = {1: 65.0, 2: 18.0, 3: 7.5}[pc]
        fare  = round(float(base * rng.lognormal(0, 0.45)), 4)
        cabin = _cabin(rng, pc)
        emb   = str(rng.choice(["S","C","Q"], p=[0.70, 0.20, 0.10]))
        ticket = _ticket(rng, pc)
        if include_survived:
            prob = _survival_prob(sex, pc, age)
            survived = int(rng.random() < prob)
            rows.append((pid, survived, pc, name, sex, age, sibsp, parch, ticket, fare, cabin, emb))
        else:
            rows.append((pid, pc, name, sex, age, sibsp, parch, ticket, fare, cabin, emb))
    return rows


def _to_dataframe(rows: list[tuple], include_survived: bool) -> pd.DataFrame:
    if include_survived:
        cols = ["PassengerId","Survived","Pclass","Name","Sex","Age",
                "SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
    else:
        cols = ["PassengerId","Pclass","Name","Sex","Age",
                "SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
    return pd.DataFrame(rows, columns=cols)


def main() -> None:
    rng = np.random.default_rng(42)
    raw_dir = Path("data/raw")
    ensure_dir(raw_dir)

    # ── Training CSV (200 rows) ───────────────────────────────────────────────
    train_seed = _to_dataframe(TRAIN_SEED_ROWS, include_survived=True)
    n_extra_train = 200 - len(train_seed)
    extra_train = _generate_rows(n_extra_train, start_id=13, rng=rng, include_survived=True)
    train_df = pd.concat(
        [train_seed, _to_dataframe(extra_train, include_survived=True)],
        ignore_index=True,
    )
    train_path = raw_dir / "titanic_train.csv"
    train_df.to_csv(train_path, index=False)
    survived_count = int(train_df["Survived"].sum())
    print(f"Training data : {len(train_df)} rows | survived {survived_count} / {len(train_df)}")
    print(f"Wrote: {train_path}")

    # ── Inference CSV (50 rows, no label) ────────────────────────────────────
    infer_seed = _to_dataframe(INFER_SEED_ROWS, include_survived=False)
    n_extra_infer = 50 - len(infer_seed)
    extra_infer = _generate_rows(n_extra_infer, start_id=201, rng=rng, include_survived=False)
    infer_df = pd.concat(
        [infer_seed, _to_dataframe(extra_infer, include_survived=False)],
        ignore_index=True,
    )
    infer_path = raw_dir / "titanic_inference.csv"
    infer_df.to_csv(infer_path, index=False)
    print(f"Inference data: {len(infer_df)} rows (unlabeled)")
    print(f"Wrote: {infer_path}")


if __name__ == "__main__":
    main()
