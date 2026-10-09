"""Sampler legalnych produkcji i dekodowanie z maską.

`sample` w każdym kroku losuje jednostajnie spośród produkcji, które parser
roli jeszcze domknie w pozostałym budżecie 64 tokenów. Ziarno powtarza ten
sam komunikat. `propose`, `use` i `solve` wchodzą dopiero po `[TRY]`;
po czterech `[TRY]` zostaje komunikat końcowy.

`speak` przy ε = 1 oddaje ten sampler. Poniżej 1, w każdym kroku produkcji,
z prawdopodobieństwem ε bierze produkcję jednostajnie, w przeciwnym razie
z logitu sieci na tokenach, które tę produkcję odróżniają.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from kanon.language import MAX_ARITY, SLOTS, ParseError, parse_message, tokenize
from kanon.roles import Role

MAX_TOKENS = 64

_ROLES = {
    "[AUTHOR]": Role.AUTHOR,
    "[BREAK]": Role.BREAKER,
    "[USE]": Role.USER,
    "[OUTSIDER]": Role.OUTSIDER,
}
_ROLE_MARKERS = frozenset(_ROLES)


@dataclass(frozen=True, slots=True)
class _Obs:
    task_arity: int
    tries: int
    canon: tuple[tuple[str, int], ...]
    extra_slot: str | None
    extra_arity: int | None


@dataclass(frozen=True, slots=True)
class _View:
    task_arity: int
    tries: int
    slot: str | None
    break_arity: int
    calls: tuple[tuple[str, int], ...]
    candidate: str | None


def sample(marker: str, observation: str, *, seed: int) -> str:
    """Komunikat, który parser danej roli przyjmuje."""
    if type(seed) is not int:
        raise ValueError("seed")
    role = _ROLES.get(marker)
    if role is None:
        raise ValueError("marker")
    view = _view_for(role, _read(observation))
    text = " ".join(_Gen(random.Random(seed)).message(role, view))
    try:
        parse_message(text, arity=view.task_arity)
    except ParseError as exc:
        raise RuntimeError(text) from exc
    return text


class _Gen:
    def __init__(self, rng: random.Random, *, agent=None, epsilon: float = 1.0) -> None:
        self.rng = rng
        self.agent = agent
        self.epsilon = epsilon
        self.limit = MAX_TOKENS
        self.used = 0
        self.allow_self = False
        self.view = _View(0, 0, None, 1, (), None)
        self.emitted: list[str] = []
        self._cache: dict[tuple[str, ...], dict[str, float]] = {}

    def message(self, role: Role, view: _View) -> list[str]:
        self.view = view
        self.allow_self = False
        self.used = 0
        self.limit = MAX_TOKENS
        self.emitted = []
        self._cache = {}
        if role is Role.AUTHOR:
            return self._pick(self._author_options())
        if role is Role.BREAKER:
            return self._pick(self._breaker_options())
        if role is Role.USER:
            return self._pick(self._user_options())
        return self._pick(self._outsider_options())

    def expr(self, *, debt: str | None) -> list[str]:
        options: list[tuple[int, Callable[[], list[str]]]] = []
        if debt is None:
            options.append((3, ("(", "zero"), lambda: self.emit("(", "zero", ")")))
            arity = self.view.task_arity
            for index in range(arity):
                options.append(
                    (4, ("(", "var", str(index)), lambda index=index: self.emit("(", "var", str(index), ")"))
                )
        child = self._min_expr(debt)
        options.append((3 + child, ("(", "succ"), lambda debt=debt: self.compound(("(", "succ"), 1, debt)))
        options.append((3 + child, ("(", "pred"), lambda debt=debt: self.compound(("(", "pred"), 1, debt)))
        if debt is None:
            if0_cost = 12
        else:
            if0_cost = 9 + self._min_expr(debt)
        options.append((if0_cost, ("(", "if0"), lambda debt=debt: self.compound(("(", "if0"), 3, debt)))
        if self.allow_self and not (debt is not None and self.view.task_arity == 0):
            count = self.view.task_arity
            if debt is None:
                self_cost = 3 + 3 * count
            else:
                self_cost = 3 + self._min_expr(debt) + 3 * (count - 1)
            options.append(
                (
                    self_cost,
                    ("(", "self"),
                    lambda debt=debt, count=count: self.compound(("(", "self"), count, debt),
                )
            )
        for slot, slot_arity in self.view.calls:
            option = self._call_option(slot, slot_arity, debt)
            if option is not None:
                options.append(option)
        return self._pick(options)

    def compound(self, head: tuple[str, ...], count: int, debt: str | None) -> list[str]:
        tokens = self.emit(*head)
        payer: int | None = None
        if debt is not None:
            if count == 0:
                raise RuntimeError("unpayable call")
            payer = 0 if count == 1 else self.rng.choice(range(count))
        for index in range(count):
            child_debt = debt if index == payer else None
            later = 1
            for later_index in range(index + 1, count):
                owed = debt if later_index == payer else None
                later += self._min_expr(owed)
            tokens += self.reserve(later, lambda child_debt=child_debt: self.expr(debt=child_debt))
        tokens += self.emit(")")
        return tokens

    def _author_options(self) -> list[tuple[int, Callable[[], list[str]]]]:
        wrapped = 3 + 5 + self._min_expr(None)
        options = []
        if self.view.tries < 4:
            options.append((wrapped, ("(", "draft"), self._draft_def))
        if self.view.tries >= 1:
            options.append((wrapped, ("(", "propose"), self._propose))
        return options

    def _draft_def(self) -> list[str]:
        return self._wrap("draft", self._definition)

    def _propose(self) -> list[str]:
        return self._wrap("propose", self._definition)

    def _definition(self) -> list[str]:
        view = self.view
        if view.slot is None:
            raise RuntimeError("free slot")
        tokens = self.emit("(", "def", view.slot, str(view.task_arity))
        self.allow_self = True
        try:
            tokens += self.reserve(1, lambda: self.expr(debt=None))
        finally:
            self.allow_self = False
        tokens += self.emit(")")
        return tokens

    def _user_options(self) -> list[tuple[int, Callable[[], list[str]]]]:
        debt = self.view.candidate
        if debt is None:
            raise RuntimeError("candidate slot")
        paying = self._min_expr(debt)
        options = []
        if self.view.tries < 4:
            options.append((3 + paying, ("(", "draft"), self._draft_call))
        if self.view.tries >= 1:
            options.append((4 + paying, ("(", "use", debt), self._use))
        return options

    def _draft_call(self) -> list[str]:
        debt = self.view.candidate
        if debt is None:
            raise RuntimeError("candidate slot")
        return self._wrap("draft", lambda debt=debt: self.expr(debt=debt))

    def _use(self) -> list[str]:
        debt = self.view.candidate
        if debt is None:
            raise RuntimeError("candidate slot")
        tokens = self.emit("(", "use", debt)
        tokens += self.reserve(1, lambda debt=debt: self.expr(debt=debt))
        tokens += self.emit(")")
        return tokens

    def _outsider_options(self) -> list[tuple[int, Callable[[], list[str]]]]:
        plain = self._min_expr(None)
        options = []
        if self.view.tries < 4:
            options.append((3 + plain, ("(", "draft"), self._draft_plain))
        if self.view.tries >= 1:
            options.append((3 + plain, ("(", "solve"), self._solve))
        return options

    def _draft_plain(self) -> list[str]:
        return self._wrap("draft", lambda: self.expr(debt=None))

    def _solve(self) -> list[str]:
        return self._wrap("solve", lambda: self.expr(debt=None))

    def _breaker_options(self) -> list[tuple[int, Callable[[], list[str]]]]:
        slot = self.view.slot
        if slot is None:
            raise RuntimeError("breaker slot")
        count = self.view.break_arity
        return [
            (4, ("(", "abstain", slot), lambda slot=slot: self.emit("(", "abstain", slot, ")")),
            (4 + 4 * count, ("(", "break", slot), self._break),
        ]

    def _break(self) -> list[str]:
        slot = self.view.slot
        if slot is None:
            raise RuntimeError("breaker slot")
        tokens = self.emit("(", "break", slot)
        count = self.view.break_arity
        for index in range(count):
            later = 1 + 4 * (count - index - 1)
            tokens += self.reserve(later, self._num)
        tokens += self.emit(")")
        return tokens

    def _num(self) -> list[str]:
        options = [(4, ("(", "num", str(k)), lambda k=k: self.emit("(", "num", str(k), ")")) for k in range(11)]
        return self._pick(options)

    def _call_option(
        self,
        slot: str,
        arity: int,
        debt: str | None,
    ) -> tuple[int, Callable[[], list[str]]] | None:
        discharges = debt is not None and slot == debt
        if debt is not None and not discharges and arity == 0:
            return None
        if discharges or debt is None:
            cost = 4 + 3 * arity
            child_debt = None
        else:
            cost = 4 + self._min_expr(debt) + 3 * (arity - 1)
            child_debt = debt

        def build(slot: str = slot, arity: int = arity, child_debt: str | None = child_debt) -> list[str]:
            return self.compound(("(", "call", slot), arity, child_debt)

        return (cost, ("(", "call", slot), build)

    def _min_expr(self, debt: str | None) -> int:
        if debt is None:
            return 3
        for slot, arity in self.view.calls:
            if slot == debt:
                return 4 + 3 * arity
        return MAX_TOKENS + 1

    def _wrap(self, head: str, inner: Callable[[], list[str]]) -> list[str]:
        tokens = self.emit("(", head)
        tokens += self.reserve(1, inner)
        tokens += self.emit(")")
        return tokens

    def _pick(self, options: list[tuple]) -> list[str]:
        room = self.limit - self.used
        legal = [option for option in options if option[0] <= room]
        if not legal:
            raise RuntimeError("no legal production")
        if self.agent is None or self.epsilon >= 1:
            return self.rng.choice(legal)[-1]()
        if len(legal) == 1 or self.rng.random() < self.epsilon:
            return self.rng.choice(legal)[-1]()
        weights = [self._mass(option[1]) for option in legal]
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(legal)[-1]()
        draw = self.rng.random() * total
        acc = 0.0
        for option, weight in zip(legal, weights):
            acc += weight
            if draw <= acc:
                return option[-1]()
        return legal[-1][-1]()

    def _mass(self, signature: tuple[str, ...]) -> float:
        context = list(self.emitted)
        mass = 1.0
        for token in signature:
            if not context:
                context.append(token)
                continue
            prob = self._prob(context, token)
            if prob <= 0:
                return 0.0
            mass *= prob
            context.append(token)
        return mass

    def _prob(self, context: list[str], token: str) -> float:
        from kanon.model import TOKEN_ID

        if token not in TOKEN_ID or any(item not in TOKEN_ID for item in context):
            return 0.0
        return self._probs(tuple(context)).get(token, 0.0)

    def _probs(self, context: tuple[str, ...]) -> dict[str, float]:
        import torch

        from kanon.model import TOKEN_ID, VOCAB

        found = self._cache.get(context)
        if found is not None:
            return found
        ids = torch.tensor([[TOKEN_ID[token] for token in context]], dtype=torch.long)
        mask = torch.ones(len(VOCAB), dtype=torch.bool)
        with torch.no_grad():
            row = self.agent.distribution(ids, mask)[0]
        found = {token: float(row[TOKEN_ID[token]].item()) for token in TOKEN_ID}
        self._cache[context] = found
        return found

    def emit(self, *tokens: str) -> list[str]:
        if self.used + len(tokens) > self.limit:
            raise RuntimeError("token limit")
        self.used += len(tokens)
        self.emitted.extend(tokens)
        return list(tokens)

    def reserve(self, count: int, build: Callable[[], list[str]]) -> list[str]:
        self.limit -= count
        try:
            return build()
        finally:
            self.limit += count


def _view_for(role: Role, obs: _Obs) -> _View:
    calls = dict(obs.canon)
    slot = obs.extra_slot
    candidate = None
    break_arity = obs.task_arity if obs.extra_arity is None else obs.extra_arity
    if role is Role.AUTHOR:
        if slot not in SLOTS:
            raise ValueError("free slot")
    elif role is Role.BREAKER:
        if slot not in SLOTS:
            raise ValueError("breaker slot")
        if break_arity < 1:
            break_arity = 1
    elif role is Role.USER:
        if slot not in SLOTS:
            raise ValueError("candidate slot")
        candidate = slot
        calls.setdefault(candidate, obs.task_arity)
    calls_sorted = tuple(sorted(calls.items(), key=lambda item: int(item[0][1:])))
    return _View(
        task_arity=obs.task_arity,
        tries=obs.tries,
        slot=slot,
        break_arity=break_arity,
        calls=calls_sorted,
        candidate=candidate,
    )


def _read(observation: str) -> _Obs:
    tokens = tokenize(observation)
    if "[CANON]" not in tokens or "[PAIRS]" not in tokens:
        raise ValueError("observation")
    canon: list[tuple[str, int]] = []
    index = tokens.index("[CANON]") + 1
    while index < len(tokens) and tokens[index] != "[PAIRS]":
        if tokens[index] == "[SEP]":
            index += 1
            continue
        if tokens[index] != "(":
            raise ValueError("observation")
        form, index = _take_form(tokens, index)
        if len(form) < 4 or form[1] != "def" or form[2] not in SLOTS:
            raise ValueError("observation")
        canon.append((form[2], int(form[3])))
    if index >= len(tokens) or tokens[index] != "[PAIRS]":
        raise ValueError("observation")
    index += 1
    num_count = 0
    while index + 1 < len(tokens) and tokens[index] == "(" and tokens[index + 1] == "num":
        _, index = _take_form(tokens, index)
        num_count += 1
    if num_count % 5 != 0:
        raise ValueError("pairs")
    task_arity = num_count // 5 - 1
    if not 0 <= task_arity <= MAX_ARITY:
        raise ValueError("arity")
    extras: list[str] = []
    while index < len(tokens) and tokens[index] not in _ROLE_MARKERS:
        if tokens[index] == "[TRY]":
            index = _skip_try(tokens, index)
            continue
        extras.append(tokens[index])
        index += 1
    extra_slot, extra_arity = _extra(extras)
    return _Obs(task_arity, tokens.count("[TRY]"), tuple(canon), extra_slot, extra_arity)


def _extra(extras: list[str]) -> tuple[str | None, int | None]:
    if not extras:
        return None, None
    if len(extras) == 1 and extras[0] in SLOTS:
        return extras[0], None
    index = 0
    found: tuple[str, int] | None = None
    while index < len(extras):
        if extras[index] != "(":
            raise ValueError("observation")
        form, index = _take_form(extras, index)
        if len(form) >= 4 and form[1] == "def" and form[2] in SLOTS:
            if found is not None:
                raise ValueError("observation")
            found = (form[2], int(form[3]))
            continue
        if len(form) >= 2 and form[1] in ("err", "zero"):
            continue
        raise ValueError("observation")
    if found is None:
        return None, None
    return found


def _skip_try(tokens: list[str], index: int) -> int:
    index += 1
    skipped = 0
    while skipped < 5 and index < len(tokens) and tokens[index] == "(":
        _, index = _take_form(tokens, index)
        skipped += 1
    return index


def _take_form(tokens: list[str], start: int) -> tuple[list[str], int]:
    depth = 0
    index = start
    while index < len(tokens):
        if tokens[index] == "(":
            depth += 1
        elif tokens[index] == ")":
            depth -= 1
            if depth == 0:
                return tokens[start : index + 1], index + 1
        index += 1
    raise ValueError("unbalanced observation")


class _Need:
    def __init__(self, options: list[str]) -> None:
        self.options = options


def speak(
    marker: str,
    observation: str,
    *,
    seed: int,
    epsilon: float = 1.0,
    agent=None,
) -> str:
    """Komunikat roli. Przy ε = 1 jest to `sample`."""
    number = _epsilon(epsilon)
    if type(seed) is not int:
        raise ValueError("seed")
    if number >= 1:
        return sample(marker, observation, seed=seed)
    if agent is None:
        raise ValueError("agent")
    role = _ROLES.get(marker)
    if role is None:
        raise ValueError("marker")
    view = _view_for(role, _read(observation))
    tokens = _Gen(random.Random(seed), agent=agent, epsilon=number).message(role, view)
    return " ".join(tokens)


def legal_tokens(marker: str, observation: str, tokens: list[str]) -> list[str] | None:
    """Następne tokeny, które jeszcze domykają komunikat. `None` znaczy koniec."""
    role = _ROLES.get(marker)
    if role is None:
        raise ValueError("marker")
    return _legal(role, _view_for(role, _read(observation)), tokens)


def _epsilon(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("epsilon")
    number = float(value)
    if not 0 <= number <= 1:
        raise ValueError("epsilon")
    return number


def _legal(role: Role, view: _View, tokens: list[str]) -> list[str] | None:
    if role is Role.AUTHOR:
        parsed = _author_message(tokens, view)
    elif role is Role.BREAKER:
        parsed = _breaker_message(tokens, view)
    elif role is Role.USER:
        parsed = _user_message(tokens, view)
    else:
        parsed = _outsider_message(tokens, view)
    if isinstance(parsed, _Need):
        return sorted(set(parsed.options))
    if parsed == len(tokens):
        return None
    return []


def _author_message(tokens: list[str], view: _View):
    cap = MAX_TOKENS
    opened = _one(tokens, 0, "(", cap)
    if not isinstance(opened, int):
        return opened
    body = 5 + _min_expr(view, None)
    rest = cap - opened
    heads = []
    if view.tries < 4 and 1 + body + 1 <= rest:
        heads.append("draft")
    if view.tries >= 1 and 1 + body + 1 <= rest:
        heads.append("propose")
    headed = _one_of(tokens, opened, heads, cap)
    if not isinstance(headed, int):
        return headed
    defined = _definition(tokens, headed, view, cap - 1)
    if not isinstance(defined, int):
        return defined
    return _close(tokens, defined, cap)


def _definition(tokens: list[str], index, view: _View, cap: int):
    if view.slot is None:
        return None
    step = _one(tokens, index, "(", cap)
    step = _one(tokens, step, "def", cap)
    step = _one(tokens, step, view.slot, cap)
    step = _one(tokens, step, str(view.task_arity), cap)
    if not isinstance(step, int):
        return step
    body = _expr(tokens, step, view, None, True, cap - 1)
    if not isinstance(body, int):
        return body
    return _close(tokens, body, cap)


def _user_message(tokens: list[str], view: _View):
    cap = MAX_TOKENS
    debt = view.candidate
    if debt is None:
        return None
    paying = _min_expr(view, debt)
    opened = _one(tokens, 0, "(", cap)
    if not isinstance(opened, int):
        return opened
    rest = cap - opened
    heads = []
    if view.tries < 4 and 2 + paying <= rest:
        heads.append("draft")
    if view.tries >= 1 and 3 + paying <= rest:
        heads.append("use")
    headed = _one_of(tokens, opened, heads, cap)
    if not isinstance(headed, int):
        return headed
    if tokens[opened] == "draft":
        body = _expr(tokens, headed, view, debt, False, cap - 1)
    else:
        slotted = _one(tokens, headed, debt, cap)
        if not isinstance(slotted, int):
            return slotted
        body = _expr(tokens, slotted, view, debt, False, cap - 1)
    if not isinstance(body, int):
        return body
    return _close(tokens, body, cap)


def _outsider_message(tokens: list[str], view: _View):
    cap = MAX_TOKENS
    plain = _min_expr(view, None)
    opened = _one(tokens, 0, "(", cap)
    if not isinstance(opened, int):
        return opened
    rest = cap - opened
    heads = []
    if view.tries < 4 and 2 + plain <= rest:
        heads.append("draft")
    if view.tries >= 1 and 2 + plain <= rest:
        heads.append("solve")
    headed = _one_of(tokens, opened, heads, cap)
    if not isinstance(headed, int):
        return headed
    body = _expr(tokens, headed, view, None, False, cap - 1)
    if not isinstance(body, int):
        return body
    return _close(tokens, body, cap)


def _breaker_message(tokens: list[str], view: _View):
    cap = MAX_TOKENS
    slot = view.slot
    if slot is None:
        return None
    count = view.break_arity
    opened = _one(tokens, 0, "(", cap)
    if not isinstance(opened, int):
        return opened
    heads = []
    if 4 <= cap:
        heads.append("abstain")
    if 4 + 4 * count <= cap:
        heads.append("break")
    headed = _one_of(tokens, opened, heads, cap)
    if not isinstance(headed, int):
        return headed
    slotted = _one(tokens, headed, slot, cap)
    if not isinstance(slotted, int):
        return slotted
    if tokens[opened] == "abstain":
        return _close(tokens, slotted, cap)
    cursor = slotted
    for index in range(count):
        later = 1 + 4 * (count - index - 1)
        parsed = _num(tokens, cursor, cap - later)
        if not isinstance(parsed, int):
            return parsed
        cursor = parsed
    return _close(tokens, cursor, cap)


def _expr(tokens, index, view: _View, debt: str | None, allow_self: bool, cap: int):
    if not isinstance(index, int):
        return index
    room = cap - index
    if room < _min_expr(view, debt):
        return None
    opened = _one(tokens, index, "(", cap)
    if not isinstance(opened, int):
        return opened
    heads = _heads(view, debt, allow_self, room)
    chosen = _one_of(tokens, opened, heads, cap)
    if not isinstance(chosen, int):
        return chosen
    head = tokens[opened]
    if head == "zero":
        return _close(tokens, chosen, cap)
    if head == "var":
        indexes = [str(item) for item in range(view.task_arity)]
        picked = _one_of(tokens, chosen, indexes, cap)
        if not isinstance(picked, int):
            return picked
        return _close(tokens, picked, cap)
    if head in ("succ", "pred"):
        return _fan(tokens, chosen, 1, debt, allow_self, view, cap)
    if head == "if0":
        return _fan(tokens, chosen, 3, debt, allow_self, view, cap)
    if head == "self":
        return _fan(tokens, chosen, view.task_arity, debt, allow_self, view, cap)
    slots = _call_slots(view, debt, room)
    slotted = _one_of(tokens, chosen, slots, cap)
    if not isinstance(slotted, int):
        return slotted
    slot = tokens[chosen]
    arity = dict(view.calls)[slot]
    discharges = debt is not None and slot == debt
    child_debt = None if discharges or debt is None else debt
    return _fan(tokens, slotted, arity, child_debt, allow_self, view, cap)


def _fan(tokens, index, count: int, debt: str | None, allow_self: bool, view: _View, cap: int):
    if debt is None:
        return _children(tokens, index, count, None, None, allow_self, view, cap)
    if count == 0:
        return None
    needs: list[str] = []
    ends: list[int] = []
    for payer in range(count):
        parsed = _children(tokens, index, count, debt, payer, allow_self, view, cap)
        if isinstance(parsed, _Need):
            needs.extend(parsed.options)
        elif isinstance(parsed, int):
            ends.append(parsed)
    if ends:
        return ends[0]
    if needs:
        return _Need(needs)
    return None


def _children(tokens, index, count, debt, payer, allow_self, view, cap):
    cursor = index
    for child in range(count):
        later = 1
        for later_index in range(child + 1, count):
            owed = debt if later_index == payer else None
            later += _min_expr(view, owed)
        child_debt = debt if child == payer else None
        parsed = _expr(tokens, cursor, view, child_debt, allow_self, cap - later)
        if not isinstance(parsed, int):
            return parsed
        cursor = parsed
    return _close(tokens, cursor, cap)


def _num(tokens, index, cap: int):
    if not isinstance(index, int) or cap - index < 4:
        return None if isinstance(index, int) else index
    step = _one(tokens, index, "(", cap)
    step = _one(tokens, step, "num", cap)
    step = _one_of(tokens, step, [str(k) for k in range(11)], cap)
    return _close(tokens, step, cap)


def _heads(view: _View, debt: str | None, allow_self: bool, room: int) -> list[str]:
    heads: list[str] = []
    paid = _min_expr(view, debt)
    if debt is None and room >= 3:
        heads.append("zero")
    if debt is None and view.task_arity > 0 and room >= 4:
        heads.append("var")
    child = paid if debt is not None else 3
    if room >= 3 + child:
        heads.extend(("succ", "pred"))
    if debt is None and room >= 12:
        heads.append("if0")
    if debt is not None and room >= 9 + paid:
        heads.append("if0")
    count = view.task_arity
    if allow_self and not (debt is not None and count == 0):
        if debt is None and room >= 3 + 3 * count:
            heads.append("self")
        if debt is not None and room >= paid + 3 * count:
            heads.append("self")
    if _call_slots(view, debt, room):
        heads.append("call")
    return heads


def _call_slots(view: _View, debt: str | None, room: int) -> list[str]:
    found = []
    for slot, arity in view.calls:
        discharges = debt is not None and slot == debt
        if debt is not None and not discharges and arity == 0:
            continue
        if discharges or debt is None:
            cost = 4 + 3 * arity
        else:
            cost = 4 + _min_expr(view, debt) + 3 * (arity - 1)
        if cost <= room:
            found.append(slot)
    return found


def _min_expr(view: _View, debt: str | None) -> int:
    if debt is None:
        return 3
    for slot, arity in view.calls:
        if slot == debt:
            return 4 + 3 * arity
    return MAX_TOKENS + 1


def _one(tokens, index, expected: str, cap: int):
    if not isinstance(index, int):
        return index
    if index >= cap:
        return None
    if index >= len(tokens):
        return _Need([expected])
    if tokens[index] != expected:
        return None
    return index + 1


def _one_of(tokens, index, options: list[str], cap: int):
    if not isinstance(index, int):
        return index
    found = list(dict.fromkeys(options))
    if not found or index >= cap:
        return None
    if index >= len(tokens):
        return _Need(found)
    if tokens[index] not in found:
        return None
    return index + 1


def _close(tokens, index, cap: int):
    return _one(tokens, index, ")", cap)
