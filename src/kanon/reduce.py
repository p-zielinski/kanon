"""Reduktor Kalk do liczby naturalnej albo błędu fuel / memory / steps.

Stos ewaluacji jest na stercie. Paliwo i limit kroków nie zależą od stosu Pythona.
`(self ...)` kosztuje 1 paliwa bieżącego wywołania. `(call ...)` startuje od 256.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from kanon.language import (
    Call,
    Def,
    Expr,
    If0,
    Num,
    ParseError,
    Pred,
    Self,
    Succ,
    Var,
    Zero,
    node_count,
)

FUEL_LIMIT = 256
NODE_LIMIT = 4096
STEP_LIMIT = 10_000


class ReduceError(Exception):
    def __init__(self, kind: str):
        if kind not in ("fuel", "memory", "steps"):
            raise ValueError(kind)
        self.kind = kind
        super().__init__(kind)


class _Fuel:
    __slots__ = ("left",)

    def __init__(self, left: int = FUEL_LIMIT):
        self.left = left


class _Budget:
    __slots__ = ("steps",)

    def __init__(self) -> None:
        self.steps = 0

    def tick(self) -> None:
        self.steps += 1
        if self.steps > STEP_LIMIT:
            raise ReduceError("steps")


def reduce_expr(
    expr: Expr,
    *,
    defs: Mapping[str, Def] | None = None,
    env: Sequence[int] = (),
) -> int:
    _guard(expr)
    return _eval(expr, tuple(env), _Fuel(), _Budget(), dict(defs or {}), None)


def reduce_def(
    defn: Def,
    args: Sequence[int | Expr],
    *,
    defs: Mapping[str, Def] | None = None,
) -> int:
    merged = dict(defs or {})
    merged[defn.slot] = defn
    _guard(defn.body)
    budget = _Budget()
    values = tuple(_arg_value(arg, budget, merged) for arg in args)
    if len(values) != defn.arity:
        raise ValueError("argument count does not match arity")
    return _eval(defn.body, values, _Fuel(), budget, merged, defn)


def _arg_value(arg: int | Expr, budget: _Budget, defs: dict[str, Def]) -> int:
    if type(arg) is int:
        if arg < 0:
            raise ValueError("negative argument")
        return arg
    _guard(arg)
    return _eval(arg, (), _Fuel(), budget, defs, None)


def _guard(expr: Expr) -> None:
    if node_count(expr) > NODE_LIMIT:
        raise ReduceError("memory")


def _eval(
    expr: Expr,
    env: tuple[int, ...],
    fuel: _Fuel,
    budget: _Budget,
    defs: dict[str, Def],
    self_def: Def | None,
) -> int:
    results: list[int] = []
    work: list[tuple] = [("eval", expr, env, fuel, self_def)]
    while work:
        item = work.pop()
        tag = item[0]
        if tag == "add":
            results.append(results.pop() + item[1])
            continue
        if tag == "pred":
            results.append(max(0, results.pop() - item[1]))
            continue
        if tag == "if0":
            _, zero_branch, nonzero_branch, env, fuel, self_def = item
            chosen = zero_branch if results.pop() == 0 else nonzero_branch
            work.append(("eval", chosen, env, fuel, self_def))
            continue
        if tag == "self-enter":
            _, defn, fuel, nargs = item
            args = _take(results, nargs)
            if fuel.left <= 0:
                raise ReduceError("fuel")
            fuel.left -= 1
            work.append(("eval", defn.body, args, fuel, defn))
            continue
        if tag == "call-enter":
            _, slot, nargs = item
            args = _take(results, nargs)
            try:
                target = defs[slot]
            except KeyError:
                raise ValueError(f"unknown slot {slot}") from None
            if len(args) != target.arity:
                raise ValueError("call arity does not match def")
            work.append(("eval", target.body, args, _Fuel(), target))
            continue
        _, expr, env, fuel, self_def = item
        budget.tick()
        if isinstance(expr, Succ):
            work.append(("add", 1))
            work.append(("eval", expr.arg, env, fuel, self_def))
            continue
        if isinstance(expr, Pred):
            work.append(("pred", 1))
            work.append(("eval", expr.arg, env, fuel, self_def))
            continue
        if isinstance(expr, Zero):
            results.append(0)
            continue
        if isinstance(expr, Num):
            if expr.value + 1 > NODE_LIMIT:
                raise ReduceError("memory")
            results.append(expr.value)
            continue
        if isinstance(expr, Var):
            try:
                results.append(env[expr.index])
            except IndexError:
                raise ValueError("var out of range") from None
            continue
        if isinstance(expr, If0):
            work.append(
                (
                    "if0",
                    expr.zero_branch,
                    expr.nonzero_branch,
                    env,
                    fuel,
                    self_def,
                )
            )
            work.append(("eval", expr.cond, env, fuel, self_def))
            continue
        if isinstance(expr, Self):
            if self_def is None:
                raise ParseError("self outside def")
            work.append(("self-enter", self_def, fuel, len(expr.args)))
            for arg in reversed(expr.args):
                work.append(("eval", arg, env, fuel, self_def))
            continue
        if isinstance(expr, Call):
            work.append(("call-enter", expr.slot, len(expr.args)))
            for arg in reversed(expr.args):
                work.append(("eval", arg, env, fuel, self_def))
            continue
        raise TypeError("not an expression")
    if len(results) != 1:
        raise RuntimeError("reducer did not leave one result")
    return results[0]


def _take(results: list[int], nargs: int) -> tuple[int, ...]:
    if nargs == 0:
        return ()
    args = tuple(results[-nargs:])
    del results[-nargs:]
    return args
