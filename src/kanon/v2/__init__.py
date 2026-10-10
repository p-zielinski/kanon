"""Kanon v2.

Ziarno rundy, słownik dnia zero, budżet komunikatu, sieć z oknem 256,
półka, przegląd do 16 tokenów, dziura prosta, piętra 2–12, paczka schematu,
kernel taktyk, filtr E, pula, warsztat poziomów 0–3, runda ćwiczenia,
taśma i ε, rampa poziomów, mutacja po zejściu z piętra 12
oraz zbiór 50 zadań po zejściu z piętra 8, metryki, snapshot, obrót zdarzeń
i atomowy `resume.pt` w `runs/v2-seed0/`, krok AdamW przy batchu 4
odczyt jednego termu albo jednego zdania,
kopia co 6 godzin poza plikiem sieci
oraz jedno polecenie `python -m kanon.v2` na wiele dni.
"""

from kanon.v2.ask import ask
from kanon.v2.batch import run_package, run_packages
from kanon.v2.budget import message_budget
from kanon.v2.command import configure, main
from kanon.v2.kernel import Kernel, Verdict
from kanon.v2.learn import BATCH, GRAD_CLIP, LR, produce, step
from kanon.v2.measure import build_set, score_set, set_seed, write_metric
from kanon.v2.mutate import (
    Gap,
    choose_fact,
    finish_package,
    is_mutation,
    mutate,
    replay,
    shared_facts,
    skeletons,
    solo_task,
)
from kanon.v2.play import Clocks, Tape, apply_step, play_exercise, run_schema, speaker
from kanon.v2.ramp import Ramp
from kanon.v2.record import Milestones, agent_observation, append_event, append_row, write_snapshot
from kanon.v2.resume import DEFAULT_DIR, Session
from kanon.v2.schema import (
    Schema,
    comparison_shape,
    decode_n1,
    decode_p,
    decode_r1,
    decode_r2,
    encode_n1,
    encode_p,
    encode_r1,
    encode_r2,
    floor_family,
    floor_shape,
    gcd_shape,
)
from kanon.v2.seed import round_seed
from kanon.v2.shape import decode_hole, decode_pair, hole_count
from kanon.v2.shelf import FrozenShelf, Shelf
from kanon.v2.ship import ShipClock, bundle, send
from kanon.v2.survey import survey, week_ready
from kanon.v2.tactics import Fact, admits_e, induction_ok, settle
from kanon.v2.vocab import VOCAB
from kanon.v2.workshop import Pending, Workshop, draw_task, ramp_ceiling

__all__ = [
    "BATCH",
    "Clocks",
    "DEFAULT_DIR",
    "Fact",
    "FrozenShelf",
    "Gap",
    "GRAD_CLIP",
    "Kernel",
    "LR",
    "Milestones",
    "Pending",
    "Ramp",
    "Schema",
    "Session",
    "Shelf",
    "ShipClock",
    "Tape",
    "VOCAB",
    "Verdict",
    "Workshop",
    "admits_e",
    "agent_observation",
    "append_event",
    "append_row",
    "apply_step",
    "ask",
    "build_set",
    "bundle",
    "choose_fact",
    "comparison_shape",
    "configure",
    "decode_hole",
    "decode_n1",
    "decode_p",
    "decode_pair",
    "decode_r1",
    "decode_r2",
    "draw_task",
    "encode_n1",
    "encode_p",
    "encode_r1",
    "encode_r2",
    "finish_package",
    "floor_family",
    "floor_shape",
    "gcd_shape",
    "hole_count",
    "induction_ok",
    "is_mutation",
    "main",
    "message_budget",
    "mutate",
    "play_exercise",
    "produce",
    "ramp_ceiling",
    "replay",
    "round_seed",
    "run_package",
    "run_packages",
    "run_schema",
    "score_set",
    "send",
    "set_seed",
    "settle",
    "shared_facts",
    "skeletons",
    "solo_task",
    "speaker",
    "step",
    "survey",
    "week_ready",
    "write_metric",
    "write_snapshot",
]
