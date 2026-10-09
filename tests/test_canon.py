import pytest

from kanon.language import Def, Var, parse_def, parse_expr
from kanon.tasks import IOPair
from kanon.canon import BreakerCase, Canon, Grave

IDENTITY = "(var 0)"


def test_admit_stores_def_with_pairs_and_round():
    canon = Canon()
    defn = _defn("f0")
    public = _public(lambda i: i)
    audit = _audit(lambda i: i)
    assert canon.admit(defn, public, audit, 4) == "f0"
    entry = canon.replay()[0]
    assert entry.slot == "f0"
    assert entry.definition == defn
    assert entry.public_pairs == public
    assert entry.audit_pairs == audit
    assert len(entry.audit_pairs) == 16
    assert entry.born_round == 4
    assert entry.last_used_round == 4


def test_admit_refuses_an_incomplete_bundle():
    canon = Canon()
    defn = _defn("f0")
    with pytest.raises(ValueError):
        canon.admit(defn, (), _audit(lambda i: i), 0)
    with pytest.raises(ValueError):
        canon.admit(defn, _public(lambda i: i), _audit(lambda i: i)[:15], 0)
    with pytest.raises(ValueError):
        canon.admit(Def("f32", 1, Var(0)), _public(lambda i: i), _audit(lambda i: i), 0)
    assert canon.replay() == ()


def test_call_in_a_solution_updates_last_used_round():
    canon = Canon()
    canon.admit(_defn("f0"), _public(lambda i: i), _audit(lambda i: i), 0)
    canon.admit(_defn("f1"), _public(lambda i: i), _audit(lambda i: i), 0)
    term = parse_expr("(if0 (zero) (call f0 (var 0)) (succ (call f1 (var 0))))", arity=1)
    canon.note_use(term, 6)
    by_slot = {entry.slot: entry for entry in canon.replay()}
    assert by_slot["f0"].last_used_round == 6
    assert by_slot["f1"].last_used_round == 6
    canon.note_use(parse_expr("(succ (var 0))", arity=1), 9)
    assert {entry.slot: entry for entry in canon.replay()}["f0"].last_used_round == 6


def test_twelve_rounds_without_a_call_frees_the_slot_and_drops_replay():
    canon = Canon()
    defn = _defn("f0")
    canon.admit(defn, _public(lambda i: i), _audit(lambda i: i), 0)
    assert canon.expire(11) == ()
    assert canon.replay()[0].definition == defn
    assert canon.expire(12) == ("f0",)
    assert canon.replay() == ()
    assert "f0" in canon.free_slots()
    assert canon.admit(_defn("f0", "(succ (var 0))"), _public(lambda i: i + 1), _audit(lambda i: i + 1), 12) == "f0"

    kept = Canon()
    kept.admit(defn, _public(lambda i: i), _audit(lambda i: i), 0)
    kept.note_use(parse_expr("(call f0 (var 0))", arity=1), 11)
    assert kept.expire(22) == ()
    assert kept.expire(23) == ("f0",)


def test_bench_rechecks_only_the_oldest_entry_every_fourth_round():
    canon = Canon()
    canon.admit(_defn("f0"), _public(lambda i: i), _audit(lambda i: i), 0)
    canon.admit(_defn("f1"), _public(lambda i: 0), _audit(lambda i: 0), 0)
    assert canon.bench(3) is None
    assert {entry.slot for entry in canon.replay()} == {"f0", "f1"}
    assert canon.bench(4) is None
    assert {entry.slot for entry in canon.replay()} == {"f0", "f1"}

    mismatched = Canon()
    mismatched.admit(_defn("f0"), _public(lambda i: 0), _audit(lambda i: 0), 0)
    mismatched.admit(_defn("f1"), _public(lambda i: 0), _audit(lambda i: 0), 0)
    assert mismatched.bench(4) == "f0"
    assert [entry.slot for entry in mismatched.replay()] == ["f1"]
    assert "f0" in mismatched.free_slots()


def test_bench_sees_a_drift_when_a_called_slot_changes_meaning():
    canon = Canon()
    caller = _defn("f0", "(call f1 (var 0))")
    canon.admit(caller, _public(lambda i: i), _audit(lambda i: i), 0)
    canon.admit(_defn("f1", "(succ (var 0))"), _public(lambda i: i + 1), _audit(lambda i: i + 1), 0)
    assert canon.bench(4) == "f0"
    assert [entry.slot for entry in canon.replay()] == ["f1"]


def test_end_round_expires_before_it_benches():
    canon = Canon()
    canon.admit(_defn("f0"), _public(lambda i: i), _audit(lambda i: i), 0)
    canon.admit(_defn("f1"), _public(lambda i: 0), _audit(lambda i: 0), 0)
    canon.note_use(parse_expr("(call f1 (var 0))", arity=1), 11)
    canon.end_round(12)
    assert canon.replay() == ()


def test_cemetery_keeps_results_and_breaker_export_omits_the_target_term():
    canon = Canon()
    definition = _defn("f2", "(succ (var 0))")
    target_term = parse_def("(def f3 1 (succ (succ (var 0))))")
    canon.bury(
        slot="f2",
        definition=definition,
        inputs=(4,),
        def_result=5,
        target_result=6,
        round_no=2,
    )
    grave = canon.graves[0]
    assert isinstance(grave, Grave)
    assert grave.slot == "f2"
    assert grave.definition == definition
    assert grave.inputs == (4,)
    assert grave.def_result == 5
    assert grave.target_result == 6
    assert grave.round_no == 2
    exported = canon.breaker_export()
    assert exported == (BreakerCase((4,), True),)
    assert set(BreakerCase.__dataclass_fields__) == {"inputs", "diverged"}
    text = repr(exported)
    assert "succ" not in text
    assert "f3" not in text
    assert target_term not in exported


def test_thirty_third_definition_is_refused_without_overwrite():
    canon = Canon()
    originals = []
    for index in range(32):
        defn = _defn(f"f{index}")
        originals.append(defn)
        assert canon.admit(defn, _public(lambda i: i), _audit(lambda i: i), 0) == f"f{index}"
    assert canon.free_slots() == ()
    intruder = _defn("f0", "(succ (var 0))")
    assert canon.admit(intruder, _public(lambda i: i + 1), _audit(lambda i: i + 1), 1) is None
    replay = canon.replay()
    assert len(replay) == 32
    assert [entry.slot for entry in replay] == [f"f{index}" for index in range(32)]
    assert replay[0].definition == originals[0]
    assert replay[0].definition != intruder
    assert all(entry.definition == original for entry, original in zip(replay, originals))


def _defn(slot: str, body: str = IDENTITY, arity: int = 1) -> Def:
    return parse_def(f"(def {slot} {arity} {body})")


def _audit(output) -> tuple[IOPair, ...]:
    return tuple(IOPair((i,), output(i)) for i in range(16))


def _public(output) -> tuple[IOPair, ...]:
    return (IOPair((0,), output(0)),)
