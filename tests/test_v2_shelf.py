import pytest

from kanon.language import Call, Succ, parse_def
from kanon.v2.shelf import FrozenShelf, Shelf


def test_accepted_entry_stays_and_the_buffer_expires_after_twelve_rounds():
    shelf = Shelf()
    kept = shelf.write(parse_def("(def f0 1 (var 0))"), round_no=0)
    shelf.note_use(kept, 5)
    assert kept not in shelf.expire(12)
    assert shelf.contains(kept)
    assert shelf.entry(kept).family == "identyczność"
    staged = shelf.stage(parse_def("(def f1 1 (zero))"), round_no=0)
    shelf.note_use(staged, 5)
    assert shelf.contains(staged)
    assert staged not in shelf.expire(11)
    assert shelf.contains(staged)
    assert staged in shelf.expire(12)
    assert not shelf.contains(staged)
    assert shelf.contains(kept)


def test_identity_is_named_and_two_succ_is_neither_identity_nor_successor():
    shelf = Shelf()
    ident = shelf.write(parse_def("(def f0 1 (var 0))"))
    assert shelf.entry(ident).family == "identyczność"
    other = shelf.write(parse_def("(def f0 1 (succ (succ (zero))))"))
    assert shelf.entry(other).family == "stała 2"
    assert shelf.entry(other).family not in ("identyczność", "następnik")


def test_citation_keeps_its_id_when_the_name_window_scrolls():
    shelf = Shelf()
    ids = [shelf.write(parse_def("(def f0 1 (var 0))")) for _ in range(5)]
    shelf.bind(ids[:4])
    assert shelf.id_at("f3") == ids[3]
    caller = shelf.write(parse_def("(def f9 1 (succ (call f3 (var 0))))"))
    assert shelf.entry(caller).citations == (ids[3],)
    assert "f3" not in shelf.entry(caller).citations
    body = shelf.entry(caller).body
    assert isinstance(body, Succ)
    assert isinstance(body.arg, Call)
    assert body.arg.slot == ids[3]
    shelf.bind(ids[1:5])
    assert shelf.id_at("f3") == ids[4]
    assert shelf.id_at("f3") != ids[3]
    assert shelf.entry(caller).citations == (ids[3],)


def test_group_and_solo_share_body_and_family_with_local_ids():
    defn = parse_def("(def f0 1 (var 0))")
    group, solo = Shelf(), Shelf()
    gid, sid = group.write(defn), solo.write(defn)
    assert group.entry(gid).body == solo.entry(sid).body
    assert group.entry(gid).family == solo.entry(sid).family == "identyczność"
    assert group.local_id("identyczność") == gid
    assert solo.local_id("identyczność") == sid
    pred = group.write(parse_def("(def f0 1 (pred (var 0)))"))
    assert group.entry(pred).family == "poprzednik"
    assert solo.local_id("poprzednik") is None
    assert group.local_id("identyczność") == gid


def test_bench_drops_a_number_and_keeps_a_fuel_limit():
    shelf = Shelf()
    good = shelf.write(parse_def("(def f0 1 (var 0))"))
    bad = shelf.write(parse_def(_if_past(10, "(zero)")))
    fuel = shelf.write(parse_def(_if_past(10, "(self (var 0))")))
    shelf.bind((bad,))
    citer = shelf.write(parse_def("(def f1 1 (call f0 (var 0)))"))
    pooled = shelf.pool_add((bad,))
    assert shelf.bench(49) == ()
    assert shelf.contains(bad) and shelf.contains(fuel) and shelf.contains(good)
    removed = set(shelf.bench(50))
    assert {bad, citer, pooled} <= removed
    assert good not in removed and fuel not in removed
    assert shelf.contains(good) and shelf.contains(fuel)
    assert not shelf.contains(bad)
    assert not shelf.contains(citer)
    assert pooled not in shelf.pool()


def test_frozen_weights_have_no_write_method():
    frozen = FrozenShelf()
    assert not hasattr(frozen, "write")


def _if_past(count: int, branch: str) -> str:
    expr = "(var 0)"
    for _ in range(count):
        expr = f"(pred {expr})"
    return f"(def f0 1 (if0 {expr} (var 0) {branch}))"
