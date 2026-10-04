"""Safe arithmetic evaluation for the ``calculator`` tool.

Deliberately dependency-free and free of ``eval``/``exec``: the expression is
parsed with :mod:`ast` and walked against an explicit allow-list of operators,
constants and functions.  That keeps the tool testable in isolation and stops
a model-authored expression from reaching the interpreter.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

__all__ = ["CalcError", "safe_eval"]

_MAX_EXPRESSION_LENGTH = 500
_MAX_DEPTH = 32
_MAX_POW_EXPONENT = 4096
_MAX_FACTORIAL = 5_000
# ~60k digits. Big enough for real arithmetic, small enough that a model
# cannot wedge the process materialising a multi-million-digit integer.
_MAX_RESULT_BITS = 200_000
_MAX_OPERAND_BITS = 10_000


class CalcError(ValueError):
    """Raised when an expression is invalid or refuses to evaluate."""


_BIN_OPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_COMPARE_OPS: dict[type, Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _factorial(n: Any) -> int:
    n = int(n)
    if n < 0 or n > _MAX_FACTORIAL:
        raise CalcError(f"factorial argument must be between 0 and {_MAX_FACTORIAL}")
    return math.factorial(n)


def _guard_exponent(exponent: Any) -> None:
    try:
        magnitude = abs(float(exponent))
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalcError("exponent must be a finite number") from exc
    if magnitude > _MAX_POW_EXPONENT:
        raise CalcError(f"exponent out of range (|exp| <= {_MAX_POW_EXPONENT})")


def _guard_operand(value: Any) -> None:
    if isinstance(value, int) and not isinstance(value, bool):
        if value.bit_length() > _MAX_OPERAND_BITS:
            raise CalcError(f"operand exceeds {_MAX_OPERAND_BITS} bits")


def _guard_result(value: Any) -> None:
    if isinstance(value, int) and not isinstance(value, bool):
        if value.bit_length() > _MAX_RESULT_BITS:
            raise CalcError("result is too large to represent")


def _safe_pow(base: Any, exponent: Any) -> float:
    _guard_exponent(exponent)
    return float(base) ** float(exponent)


_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "cbrt": lambda x: float(x) ** (1.0 / 3.0),
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "hypot": math.hypot,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "ln": math.log,
    "pow": _safe_pow,
    "min": min,
    "max": max,
    "sum": lambda xs: sum(xs),
    "factorial": _factorial,
    "degrees": math.degrees,
    "radians": math.radians,
    "sign": lambda x: (x > 0) - (x < 0),
}

_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "infinity": math.inf,
    "true": True,
    "false": False,
}


class _Evaluator(ast.NodeVisitor):
    def __init__(self) -> None:
        self._depth = 0

    def visit(self, node: ast.AST) -> Any:  # noqa: D102 - guards recursion
        self._depth += 1
        if self._depth > _MAX_DEPTH:
            raise CalcError("expression nests too deeply")
        try:
            return super().visit(node)
        finally:
            self._depth -= 1

    # -- leaves ---------------------------------------------------------
    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        value = node.value
        if isinstance(value, bool) or isinstance(value, (int, float)):
            return value
        raise CalcError(f"unsupported literal: {value!r}")

    def visit_Name(self, node: ast.Name) -> Any:
        key = node.id.lower()
        if key in _CONSTANTS:
            return _CONSTANTS[key]
        raise CalcError(f"unknown name: {node.id!r}")

    def visit_List(self, node: ast.List) -> Any:
        return [self.visit(elt) for elt in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(elt) for elt in node.elts)

    # -- structure ------------------------------------------------------
    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise CalcError(f"unsupported operator: {type(node.op).__name__}")

        left = self.visit(node.left)
        right = self.visit(node.right)
        _guard_operand(left)
        _guard_operand(right)
        if isinstance(node.op, ast.Pow):
            # 9 ** 99999 would otherwise be computed before anyone notices.
            _guard_exponent(right)

        try:
            return op(left, right)
        except ZeroDivisionError as exc:
            raise CalcError("division by zero") from exc
        except (OverflowError, ValueError) as exc:
            raise CalcError(str(exc)) from exc

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalcError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(self.visit(node.operand))

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        values = [self.visit(v) for v in node.values]
        if isinstance(node.op, ast.And):
            result: Any = values[0]
            for v in values[1:]:
                result = result and v
            return result
        result = values[0]
        for v in values[1:]:
            result = result or v
        return result

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _COMPARE_OPS.get(type(op_node))
            if op is None:
                raise CalcError(f"unsupported comparison: {type(op_node).__name__}")
            right = self.visit(comparator)
            if not op(left, right):
                return False
            left = right
        return True

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise CalcError("only direct function calls are allowed")
        fn = _FUNCTIONS.get(node.func.id.lower())
        if fn is None:
            raise CalcError(f"unknown function: {node.func.id}()")
        args = [self.visit(a) for a in node.args]
        kwargs = {
            kw.arg: self.visit(kw.value)
            for kw in node.keywords
            if kw.arg is not None
        }
        if any(kw.arg is None for kw in node.keywords):
            raise CalcError("**kwargs is not supported")
        try:
            return fn(*args, **kwargs)
        except CalcError:
            raise
        except RecursionError as exc:  # pragma: no cover - defensive
            raise CalcError("expression nests too deeply") from exc
        except ZeroDivisionError as exc:
            raise CalcError("division by zero") from exc
        except (ValueError, OverflowError) as exc:
            raise CalcError(str(exc)) from exc

    def generic_visit(self, node: ast.AST) -> Any:
        raise CalcError(f"unsupported syntax: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """Evaluate *expression* and return a number.

    Raises :class:`CalcError` for anything the allow-list rejects, and for
    arithmetic problems such as division by zero.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise CalcError("empty expression")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise CalcError(f"expression longer than {_MAX_EXPRESSION_LENGTH} characters")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"could not parse expression: {exc.msg}") from exc

    result = _Evaluator().visit(tree)
    if isinstance(result, bool):
        return result
    if isinstance(result, (int, float)):
        _guard_result(result)
        if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
            if math.isnan(result):
                raise CalcError("result is not a number")
            raise CalcError("result is infinite")
        return result
    raise CalcError(f"expression did not evaluate to a number (got {type(result).__name__})")
