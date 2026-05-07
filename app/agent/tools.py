from __future__ import annotations

import ast
import operator as op


class ToolError(Exception):
    pass


_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def calculator(expression: str) -> float:
    """
    Safe calculator.

    Design choice: parse with AST and allow only basic arithmetic nodes.
    """

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return float(_OPS[type(node.op)](_eval(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return float(_OPS[type(node.op)](_eval(node.left), _eval(node.right)))
        raise ToolError("Unsupported expression")

    try:
        parsed = ast.parse(expression, mode="eval")
        return _eval(parsed.body)  # type: ignore[arg-type]
    except ToolError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ToolError("Invalid expression") from e

