from kanon.language import Call, Expr, Succ, Zero, parse_def
from kanon.roles import assign, format_expr
from kanon.v2.play import (
    Tape,
    apply_step,
    encode_message,
    play_exercise,
    run_schema,
    speaker,
)
from kanon.v2.schema import Schema
from kanon.v2.shelf import Shelf
from kanon.v2.workshop import Task, draw_task

_SUCC = "(def f0 1 (succ (var 0)))"
_ADD = "(def f0 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))"
_ZERO = "(def f0 1 (zero))"
_BREAK = "(break f0 (num 0))"
_ABSTAIN = "(abstain f0)"
_KERNEL = "(claim (eq (zero) (succ (zero))) (eval))"
_MISS = "(succ (succ (succ (succ (succ (var 0))))))"


def test_one_speaker_copies_a_hit_and_the_shelf_stays():
    shelf = _shelf()
    before = tuple(card.id for card in shelf.entries())
    task = _task(shelf, level=0, round_no=0)
    hit = _reply(task)
    group = Tape()
    solo = Tape()
    clocks = _clocks()
    admitted: list[str] = []
    result = play_exercise(
        shelf,
        task,
        round_no=0,
        message=hit,
        tape=group,
        solo_tape=solo,
        solo_message=hit,
        clocks=clocks,
        admit=lambda *_args, **_kwargs: admitted.append("admit"),
    )
    assert result.attempts == 1
    assert result.solo_attempts == 1
    assert result.frozen_attempts == 1
    assert (result.speaker, result.role) == speaker(0) == (assign(0).author, "author")
    assert result.hit is True
    assert result.solo_hit is True
    assert admitted == []
    assert tuple(card.id for card in shelf.entries()) == before
    trace = group.traces(0, task.level)
    assert len(trace) == 1
    assert len(task.observation.tokens) == 176
    assert trace[0][:176] == task.observation.tokens
    assert trace[0][176:] == encode_message(hit)
    assert group.traces(1, task.level) == trace
    assert group.traces(2, task.level) == trace
    assert group.traces(3, task.level) == trace
    assert solo.traces(0, task.level) == trace
    assert solo.traces(1, task.level) == ()
    breaker = play_exercise(
        shelf,
        task,
        round_no=1,
        message=_BREAK,
        tape=Tape(),
        clocks=_clocks(),
    )
    assert breaker.attempts == 1
    assert (breaker.speaker, breaker.role) == speaker(1)
    assert speaker(1) == (assign(1).breaker, "breaker")
    assert speaker(2) == (assign(2).user, "user")
    assert speaker(3) == (assign(3).outsider, "outsider")


def test_misses_and_the_kernel_stay_off_the_tape():
    shelf = _shelf()
    task = _task(shelf, level=0, round_no=3)
    group = Tape()
    solo = Tape()
    for message in (_MISS, _ABSTAIN, _BREAK, _KERNEL):
        result = play_exercise(
            shelf,
            task,
            round_no=3,
            message=message,
            tape=group,
            solo_tape=solo,
            solo_message=message,
            clocks=_clocks(),
        )
        assert result.hit is False
        assert result.solo_hit is False
        assert result.attempts == 1
    assert group.traces(0, task.level) == ()
    assert group.traces(assign(3).outsider, task.level) == ()
    assert solo.traces(0, task.level) == ()


def test_the_257th_trace_drops_the_oldest_for_that_agent_and_level():
    shelf = _shelf()
    task = _task(shelf, level=0, round_no=0)
    hit = _reply(task)
    tape = Tape()
    for index in range(256):
        tape.note(0, task.level, (index,))
    tape.note(0, task.level + 1, (99,))
    play_exercise(
        shelf,
        task,
        round_no=0,
        message=hit,
        tape=tape,
        clocks=_clocks(),
    )
    kept = tape.traces(0, task.level)
    assert len(kept) == 256
    assert kept[0] == (1,)
    assert kept[-1][:176] == task.observation.tokens
    assert kept[-1][176:] == encode_message(hit)
    assert tape.traces(0, task.level + 1) == ((99,),)
    fresh = tape.traces(1, task.level)
    assert fresh == (kept[-1],)
    assert tape.traces(2, task.level) == fresh
    assert tape.traces(3, task.level) == fresh


def test_epsilon_ticks_one_clock_and_a_package_leaves_it_at_one():
    clocks = _clocks()
    run_schema(Schema(), _Shape(), clocks=clocks, closed=False)
    assert clocks.author(0) == 1
    assert clocks.executor(0) == 1
    assert clocks.breaker(0) == 1

    shelf = _shelf()
    task = _task(shelf, level=0, round_no=0)
    higher = _task(shelf, level=1, round_no=4)
    hit = _reply(task)
    author = _clocks()
    play_exercise(
        shelf,
        task,
        round_no=0,
        message=_ABSTAIN,
        tape=Tape(),
        clocks=author,
    )
    assert author.author(0) == 1 * 0.995
    assert author.executor(0) == 1
    assert author.author(1) == 1
    for _ in range(300):
        play_exercise(
            shelf,
            task,
            round_no=0,
            message=_ABSTAIN,
            tape=Tape(),
            clocks=author,
        )
    assert author.author(0) == 0.25
    play_exercise(
        shelf,
        higher,
        round_no=4,
        message=_ABSTAIN,
        tape=Tape(),
        clocks=author,
    )
    assert author.author(1) == 1 * 0.995
    assert author.author(0) == 0.25

    worker = _clocks()
    play_exercise(
        shelf,
        task,
        round_no=1,
        message=_ABSTAIN,
        tape=Tape(),
        clocks=worker,
    )
    assert worker.executor(0) == 1
    assert worker.author(0) == 1
    assert worker.breaker(0) == 1
    expected = 1.0
    for round_no, _role in ((1, "breaker"), (2, "user"), (3, "outsider")):
        result = play_exercise(
            shelf,
            task,
            round_no=round_no,
            message=hit,
            tape=Tape(),
            clocks=worker,
        )
        assert result.role == _role
        expected *= 0.995
        assert worker.executor(0) == expected
        assert worker.author(0) == 1
        assert worker.breaker(0) == 1
    play_exercise(
        shelf,
        task,
        round_no=1,
        message=_MISS,
        tape=Tape(),
        clocks=worker,
    )
    assert worker.executor(0) == expected
    floor = _clocks()
    for _ in range(650):
        play_exercise(
            shelf,
            task,
            round_no=1,
            message=hit,
            tape=Tape(),
            clocks=floor,
        )
    assert floor.executor(0) == 0.05
    assert floor.author(0) == 1
    assert floor.breaker(0) == 1


def test_empty_tape_skips_the_step_and_frozen_weights_stay():
    shelf = _shelf()
    task = _task(shelf, level=0, round_no=0)
    hit = _reply(task)
    group = [Bits(b"g0"), Bits(b"g1"), Bits(b"g2"), Bits(b"g3")]
    frozen = Bits(b"frozen")
    images = [item.image() for item in group]
    frozen_image = frozen.image()
    tape = Tape()
    solo = Tape()

    def step(bits: Bits) -> None:
        bits.raw = b"moved"

    missed = play_exercise(
        shelf,
        task,
        round_no=2,
        message=_ABSTAIN,
        tape=tape,
        solo_tape=solo,
        solo_message=hit,
        clocks=_clocks(),
        weights=group,
        step=step,
        frozen=frozen,
        frozen_message=hit,
    )
    assert missed.role == "user"
    assert missed.attempts == 1
    assert missed.solo_attempts == 1
    assert missed.frozen_attempts == 1
    assert missed.frozen_hit is True
    assert missed.solo_hit is True
    assert [item.image() for item in group] == images
    assert frozen.image() == frozen_image
    assert tape.traces(0, task.level) == ()
    assert solo.traces(0, task.level) == (task.observation.tokens + encode_message(hit),)
    assert apply_step(group[0], tape, agent=0, level=task.level, step=step) is False
    assert group[0].image() == images[0]

    play_exercise(
        shelf,
        task,
        round_no=0,
        message=hit,
        tape=tape,
        clocks=_clocks(),
        weights=group,
        step=step,
        frozen=frozen,
        frozen_message=hit,
    )
    assert all(item.image() == b"moved" for item in group)
    assert frozen.image() == frozen_image
    assert tape.traces(4, task.level) == ()


class Bits:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def image(self) -> bytes:
        return self.raw


class _Shape:
    name = "R1"
    count = 4

    def decode(self, index: int):
        return Zero()


def _clocks():
    from kanon.v2.play import Clocks

    return Clocks()


def _shelf() -> Shelf:
    shelf = Shelf()
    shelf.write(parse_def(_SUCC))
    shelf.write(parse_def(_ADD))
    shelf.write(parse_def(_ZERO))
    return shelf


def _task(shelf: Shelf, *, level: int, round_no: int) -> Task:
    task = draw_task(shelf, level=level, round_no=round_no, seed=0)
    assert isinstance(task, Task)
    return task


def _reply(task: Task) -> str:
    window = {task.cards[0]: "f0", task.cards[1]: "f1"}
    return format_expr(_rename(task.term, window))


def _rename(expr: Expr, window: dict[str, str]) -> Expr:
    if isinstance(expr, Call):
        slot = window.get(expr.slot, expr.slot)
        return Call(slot, tuple(_rename(arg, window) for arg in expr.args))
    if isinstance(expr, Succ):
        return Succ(_rename(expr.arg, window))
    return expr
