from __future__ import annotations

from typing import Callable, Sequence
from .const import GGML_QUANT_SIZES, GGMLQuantizationType

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Callable
from collections import deque

from numpy.typing import DTypeLike
import numpy as np

import logging
logger = logging.getLogger(__name__)

class LazyMeta(ABCMeta):

    def __new__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs):
        def __getattr__(self, name: str) -> Any:
            meta_attr = getattr(self._meta, name)
            if callable(meta_attr):
                return type(self)._wrap_fn(
                    (lambda s, *args, **kwargs: getattr(s, name)(*args, **kwargs)),
                    use_self=self,
                )
            elif isinstance(meta_attr, self._tensor_type):
                # e.g. self.T with torch.Tensor should still be wrapped
                return type(self)._wrap_fn(lambda s: getattr(s, name))(self)
            else:
                # no need to wrap non-tensor properties,
                # and they likely don't depend on the actual contents of the tensor
                return meta_attr

        namespace["__getattr__"] = __getattr__

        # need to make a builder for the wrapped wrapper to copy the name,
        # or else it fails with very cryptic error messages,
        # because somehow the same string would end up in every closures
        def mk_wrap(op_name: str, *, meta_noop: bool = False):
            # need to wrap the wrapper to get self
            def wrapped_special_op(self, *args, **kwargs):
                return type(self)._wrap_fn(
                    getattr(type(self)._tensor_type, op_name),
                    meta_noop=meta_noop,
                )(self, *args, **kwargs)
            return wrapped_special_op

        # special methods bypass __getattr__, so they need to be added manually
        # ref: https://docs.python.org/3/reference/datamodel.html#special-lookup
        # NOTE: doing this from a metaclass is very convenient
        # TODO: make this even more comprehensive
        for binary_op in (
            "lt", "le", "eq", "ne", "ge", "gt", "not"
            "abs", "add", "and", "floordiv", "invert", "lshift", "mod", "mul", "matmul",
            "neg", "or", "pos", "pow", "rshift", "sub", "truediv", "xor",
            "iadd", "iand", "ifloordiv", "ilshift", "imod", "imul", "ior", "irshift", "isub", "ixor",
            "radd", "rand", "rfloordiv", "rmul", "ror", "rpow", "rsub", "rtruediv", "rxor",
        ):
            attr_name = f"__{binary_op}__"
            # the result of these operators usually has the same shape and dtype as the input,
            # so evaluation on the meta tensor can be skipped.
            namespace[attr_name] = mk_wrap(attr_name, meta_noop=True)

        for special_op in (
            "getitem", "setitem", "len",
        ):
            attr_name = f"__{special_op}__"
            namespace[attr_name] = mk_wrap(attr_name, meta_noop=False)

        return super().__new__(cls, name, bases, namespace, **kwargs)

# Tree of lazy tensors
class LazyBase(ABC, metaclass=LazyMeta):
    _tensor_type: type
    _meta: Any
    _data: Any | None
    _lazy: deque[LazyBase]  # shared within a graph, to avoid deep recursion when making eager
    _args: tuple
    _func: Callable[[tuple], Any] | None

    def __init__(self, *, meta: Any, data: Any | None = None, lazy: deque[LazyBase] | None = None, args: tuple = (), func: Callable[[tuple], Any] | None = None):
        super().__init__()
        self._meta = meta
        self._data = data
        self._lazy = lazy if lazy is not None else deque()
        self._args = args
        self._func = func
        assert self._func is not None or self._data is not None
        if self._data is None:
            self._lazy.append(self)

    def __init_subclass__(cls) -> None:
        if "_tensor_type" not in cls.__dict__:
            raise TypeError(f"property '_tensor_type' must be defined for {cls!r}")
        return super().__init_subclass__()

    @staticmethod
    def _recurse_apply(o: Any, fn: Callable[[Any], Any]) -> Any:
        # TODO: dict and set
        if isinstance(o, (list, tuple)):
            L = []
            for item in o:
                L.append(LazyBase._recurse_apply(item, fn))
            if isinstance(o, tuple):
                L = tuple(L)
            return L
        elif isinstance(o, LazyBase):
            return fn(o)
        else:
            return o

    @classmethod
    def _wrap_fn(cls, fn: Callable, *, use_self: LazyBase | None = None, meta_noop: bool | DTypeLike | tuple[DTypeLike, Callable[[tuple[int, ...]], tuple[int, ...]]] = False) -> Callable[[Any], Any]:
        def wrapped_fn(*args, **kwargs):
            if kwargs is None:
                kwargs = {}
            args = ((use_self,) if use_self is not None else ()) + args

            meta_args = LazyBase._recurse_apply(args, lambda t: t._meta)

            if isinstance(meta_noop, bool) and not meta_noop:
                try:
                    res = fn(*meta_args, **kwargs)
                except NotImplementedError:
                    # running some operations on PyTorch's Meta tensors can cause this exception
                    res = None
            else:
                # some operators don't need to actually run on the meta tensors
                assert len(args) > 0
                res = args[0]
                assert isinstance(res, cls)
                res = res._meta
                # allow operations to override the dtype and shape
                if meta_noop is not True:
                    if isinstance(meta_noop, tuple):
                        dtype, shape = meta_noop
                        assert callable(shape)
                        res = cls.meta_with_dtype_and_shape(dtype, shape(res.shape))
                    else:
                        res = cls.meta_with_dtype_and_shape(meta_noop, res.shape)

            if isinstance(res, cls._tensor_type):
                class CollectSharedLazy:
                    # emulating a static variable
                    shared_lazy: None | deque[LazyBase] = None

                    @staticmethod
                    def collect_replace(t: LazyBase):
                        if CollectSharedLazy.shared_lazy is None:
                            CollectSharedLazy.shared_lazy = t._lazy
                        else:
                            CollectSharedLazy.shared_lazy.extend(t._lazy)
                            t._lazy = CollectSharedLazy.shared_lazy

                LazyBase._recurse_apply(args, CollectSharedLazy.collect_replace)

                shared_lazy = CollectSharedLazy.shared_lazy

                return cls(meta=cls.eager_to_meta(res), lazy=shared_lazy, args=args, func=lambda a: fn(*a, **kwargs))
            else:
                del res  # not needed
                # non-tensor return likely relies on the contents of the args
                # (e.g. the result of torch.equal)
                eager_args = cls.to_eager(args)
                return fn(*eager_args, **kwargs)
        return wrapped_fn

    @classmethod
    def to_eager(cls, t: Any) -> Any:
        def simple_to_eager(_t: LazyBase) -> Any:
            def already_eager_to_eager(_t: LazyBase) -> Any:
                assert _t._data is not None
                return _t._data

            while _t._data is None:
                lt = _t._lazy.popleft()
                if lt._data is not None:
                    # Lazy tensor did not belong in the lazy queue.
                    # Weirdly only happens with Bloom models...
                    # likely because tensors aren't unique in the queue.
                    # The final output is still the same as in eager mode,
                    # so it's safe to ignore this.
                    continue
                assert lt._func is not None
                lt._args = cls._recurse_apply(lt._args, already_eager_to_eager)
                lt._data = lt._func(lt._args)
                # sanity check
                assert lt._data is not None
                assert lt._data.dtype == lt._meta.dtype
                assert lt._data.shape == lt._meta.shape

            return _t._data

        # recurse into lists and/or tuples, keeping their structure
        return cls._recurse_apply(t, simple_to_eager)

    @classmethod
    def eager_to_meta(cls, t: Any) -> Any:
        return cls.meta_with_dtype_and_shape(t.dtype, t.shape)

    # must be overridden, meta tensor init is backend-specific
    @classmethod
    @abstractmethod
    def meta_with_dtype_and_shape(cls, dtype: Any, shape: Any) -> Any: pass

    @classmethod
    def from_eager(cls, t: Any) -> Any:
        if type(t) is cls:
            # already eager
            return t
        elif isinstance(t, cls._tensor_type):
            return cls(meta=cls.eager_to_meta(t), data=t)
        else:
            return TypeError(f"{type(t)!r} is not compatible with {cls._tensor_type!r}")

class LazyNumpyTensor(LazyBase):
    _tensor_type = np.ndarray

    @classmethod
    def meta_with_dtype_and_shape(cls, dtype: DTypeLike, shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
        # The initial idea was to use np.nan as the fill value,
        # but non-float types like np.int16 can't use that.
        # So zero it is.
        cheat = np.zeros(1, dtype)
        return np.lib.stride_tricks.as_strided(cheat, shape, (0 for _ in shape))

    def astype(self, dtype, *args, **kwargs):
        meta = type(self).meta_with_dtype_and_shape(dtype, self._meta.shape)
        full_args = (self, dtype,) + args
        # very important to pass the shared _lazy deque, or else there's an infinite loop somewhere.
        return type(self)(meta=meta, args=full_args, lazy=self._lazy, func=(lambda a: a[0].astype(*a[1:], **kwargs)))

    def tofile(self, *args, **kwargs):
        eager = LazyNumpyTensor.to_eager(self)
        return eager.tofile(*args, **kwargs)

    # TODO: __array_function__

def quant_shape_to_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType):
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % block_size != 0:
        raise ValueError(f"Quantized tensor row size ({shape[-1]}) is not a multiple of {quant_type.name} block size ({block_size})")
    return (*shape[:-1], shape[-1] // block_size * type_size)

def quant_shape_from_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType):
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % type_size != 0:
        raise ValueError(f"Quantized tensor bytes per row ({shape[-1]}) is not a multiple of {quant_type.name} type size ({type_size})")
    return (*shape[:-1], shape[-1] // type_size * block_size)

# same as ggml_compute_fp32_to_bf16 in ggml-impl.h
def __compute_fp32_to_bf16(n: np.ndarray) -> np.ndarray:
    n = n.astype(np.float32, copy=False).view(np.int32)
    # force nan to quiet
    n = np.where((n & 0x7fffffff) > 0x7f800000, (n & 0xffff0000) | (64 << 16), n)
    # flush subnormals to zero
    n = np.where((n & 0x7f800000) == 0, n & 0x80000000, n)
    # round to nearest even
    n = (n + (0x7fff + ((n >> 16) & 1))) >> 16
    return n.astype(np.int16)

# This is faster than np.vectorize and np.apply_along_axis because it works on more than one row at a time
def __apply_over_grouped_rows(func: Callable[[np.ndarray], np.ndarray], arr: np.ndarray, otype: DTypeLike, oshape: tuple[int, ...]) -> np.ndarray:
    rows = arr.reshape((-1, arr.shape[-1]))
    osize = 1
    for dim in oshape:
        osize *= dim
    out = np.empty(shape=osize, dtype=otype)
    # compute over groups of 16 rows (arbitrary, but seems good for performance)
    n_groups = rows.shape[0] // 16
    np.concatenate([func(group).ravel() for group in np.array_split(rows, n_groups)], axis=0, out=out)
    return out.reshape(oshape)

def __quantize_bf16_array(n: np.ndarray) -> np.ndarray:
    return __apply_over_grouped_rows(__compute_fp32_to_bf16, arr=n, otype=np.int16, oshape=n.shape)

__quantize_bf16_lazy = LazyNumpyTensor._wrap_fn(__quantize_bf16_array, meta_noop=np.int16)

def quantize_bf16(n: np.ndarray):
    if type(n) is LazyNumpyTensor:
        return __quantize_bf16_lazy(n)
    else:
        return __quantize_bf16_array(n)

__q8_block_size, __q8_type_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q8_0]

def can_quantize_to_q8_0(n: np.ndarray) -> bool:
    return n.shape[-1] % __q8_block_size == 0

# round away from zero
# ref: https://stackoverflow.com/a/59143326/22827863
def np_roundf(n: np.ndarray) -> np.ndarray:
    a = abs(n)
    floored = np.floor(a)
    b = floored + np.floor(2 * (a - floored))
    return np.sign(n) * b

def __quantize_q8_0_shape_change(s: tuple[int, ...]) -> tuple[int, ...]:
    return (*s[:-1], s[-1] // __q8_block_size * __q8_type_size)

# Implementation of Q8_0 with bit-exact same results as reference implementation in ggml-quants.c
def __quantize_q8_0_rows(n: np.ndarray) -> np.ndarray:
    shape = n.shape
    assert shape[-1] % __q8_block_size == 0

    n_blocks = n.size // __q8_block_size

    blocks = n.reshape((n_blocks, __q8_block_size)).astype(np.float32, copy=False)

    d = abs(blocks).max(axis=1, keepdims=True) / 127
    with np.errstate(divide="ignore"):
        id = np.where(d == 0, 0, 1 / d)
    qs = np_roundf(blocks * id)

    # (n_blocks, 2)
    d = d.astype(np.float16).view(np.uint8)
    # (n_blocks, block_size)
    qs = qs.astype(np.int8).view(np.uint8)

    assert d.shape[1] + qs.shape[1] == __q8_type_size

    return np.concatenate([d, qs], axis=1).reshape(__quantize_q8_0_shape_change(shape))

def __quantize_q8_0_array(n: np.ndarray) -> np.ndarray:
    return __apply_over_grouped_rows(__quantize_q8_0_rows, arr=n, otype=np.uint8, oshape=__quantize_q8_0_shape_change(n.shape))

__quantize_q8_0_lazy = LazyNumpyTensor._wrap_fn(
    __quantize_q8_0_array,
    meta_noop=(np.uint8, __quantize_q8_0_shape_change),
)

def quantize_q8_0(data: np.ndarray):
    if type(data) is LazyNumpyTensor:
        return __quantize_q8_0_lazy(data)
    else:
        return __quantize_q8_0_array(data)
