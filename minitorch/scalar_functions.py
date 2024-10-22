from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to the given scalar values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the addition operation.

        Args:
        ----
            ctx (Context): The context object to store information for backward computation.
            a (float): The first operand.
            b (float): The second operand.

        Returns:
        -------
            float: The result of adding a and b.

        """
        ctx.save_for_backward(a, b)
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient of the output with respect to the input.

        Args:
        ----
            ctx (Context): The context object containing saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradient of the input.

        """
        # a, b = ctx.saved_tensors
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the natural logarithm of a given input.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The natural logarithm of the input value.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the log function with respect to its input.

        Args:
        ----
            ctx (Context): The context containing saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient of the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplying two scalars.

        Args:
        ----
            ctx (Context): The context to save information for backward pass.
            a (float): The first scalar.
            b (float): The second scalar.

        Returns:
        -------
            float: The result of multiplying a and b.

        """
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradient of the multiplication operation.

        Args:
        ----
            ctx (Context): The context containing saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to the inputs (a, b).

        """
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the inverse function.

        Args:
        ----
            ctx (Context): The context to save values for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of 1 / a.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the inverse function.

        Args:
        ----
            ctx (Context): The context containing saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to the input a.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the negation function.

        Args:
        ----
            ctx (Context): The context to save values for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of -a.

        """
        return -1.0 * a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the negation function.

        Args:
        ----
            ctx (Context): The context containing saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to the input a.

        """
        return -1.0 * d_output


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the ReLU function.

        Args:
        ----
            ctx (Context): The context to save values for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of the ReLU function.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the ReLU function.

        Args:
        ----
            ctx (Context): The context containing saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to the input a.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the exponential function.

        Args:
        ----
            ctx (Context): The context to save values for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of the exponential function.

        """
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the exponential function.

        Args:
        ----
            ctx (Context): The context containing saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to the input a.

        """
        (a,) = ctx.saved_values
        return operators.exp(a) * d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the sigmoid function.

        Args:
        ----
            ctx (Context): The context to save values for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of the sigmoid function.

        """
        result = operators.sigmoid(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the sigmoid function.

        Args:
        ----
            ctx (Context): The context containing saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to the input a.

        """
        (sig,) = ctx.saved_values
        return d_output * sig * (1 - sig)


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass of the less than function.

        Args:
        ----
            ctx (Context): The context to save values for backward computation.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the less than comparison (1.0 if a < b, else 0.0).

        """
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradient of the less than function.

        Args:
        ----
            ctx (Context): The context containing saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to the inputs a and b (both 0.0).

        """
        # No gradients for comparison functions, they are non-differentiable
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass of the equal function.

        Args:
        ----
            ctx (Context): The context to save values for backward computation.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the equal comparison (1.0 if a == b, else 0.0).

        """
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradient of the equal function.

        Args:
        ----
            ctx (Context): The context containing saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to the inputs a and b (both 0.0).

        """
        return 0.0, 0.0
