"""Execute a PTX file using JSON request and result files."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
from contextlib import ExitStack
from pathlib import Path


def validate_param(param: dict) -> None:
    kind = param["type"]
    if kind not in ("i64", "f64", "i64[]", "f64[]"):
        msg = f"Unsupported parameter type: {kind}"
        raise ValueError(msg)
    values = param["values"] if kind.endswith("[]") else [param["value"]]
    if not isinstance(values, list) or not values:
        msg = "Array parameters must contain a nonempty values list"
        raise ValueError(msg)
    for value in values:
        if kind.startswith("i64"):
            valid = type(value) is int and -(2**63) <= value < 2**63
        else:
            valid = type(value) in (int, float) and math.isfinite(value)
        if not valid:
            msg = f"Invalid {kind} value: {value}"
            raise ValueError(msg)


def validate_request(request: dict) -> None:
    """Validate the wire protocol before making any CUDA calls."""
    kernel = request["kernel"]
    if not isinstance(kernel, str) or not kernel or "\0" in kernel:
        msg = "kernel must be a nonempty name without NUL bytes"
        raise ValueError(msg)
    for name in ("grid", "block"):
        dims = request.get(name, [1, 1, 1])
        if (
            not isinstance(dims, list)
            or len(dims) != 3  # noqa: PLR2004 - CUDA uses three-dimensional launches
            or any(type(d) is not int or not 0 < d < 2**32 for d in dims)
        ):
            msg = f"{name} must contain three positive uint32 dimensions"
            raise ValueError(msg)
    params = request.get("params", [])
    if not isinstance(params, list):
        msg = "params must be a list"
        raise TypeError(msg)
    for param in params:
        validate_param(param)
    validate_outputs(request.get("outputs", []), params)


def validate_outputs(outputs: list, params: list) -> None:
    if not isinstance(outputs, list):
        msg = "outputs must be a list"
        raise TypeError(msg)
    for output in outputs:
        index = output["param"]
        if type(index) is not int or not 0 <= index < len(params):
            msg = f"Output parameter index out of range: {index}"
            raise ValueError(msg)
        param = params[index]
        if not param["type"].endswith("[]"):
            msg = f"Output parameter {index} is a scalar"
            raise ValueError(msg)
        count = output.get("count", len(param["values"]))
        if type(count) is not int or not 0 <= count <= len(param["values"]):
            msg = f"Output count out of range for parameter {index}: {count}"
            raise ValueError(msg)


def execute(ptx_path: Path, request: dict) -> dict:
    validate_request(request)
    # warpforthc emits NUL-terminated PTX; accept both terminated and plain text.
    ptx = ptx_path.read_bytes().rstrip(b"\0")
    if not ptx or b"\0" in ptx:
        msg = "PTX file must contain nonempty PTX without embedded NUL bytes"
        raise ValueError(msg)
    # Import here so missing bindings/driver libraries also produce JSON errors.
    from cuda.bindings import driver  # noqa: PLC0415

    def check(result: tuple) -> object:
        error, *values = result
        if error != driver.CUresult.CUDA_SUCCESS:
            _, name = driver.cuGetErrorName(error)
            _, description = driver.cuGetErrorString(error)
            msg = f"{name.decode()}: {description.decode()}"
            raise RuntimeError(msg)
        return values[0] if values else None

    check(driver.cuInit(0))
    device = check(driver.cuDeviceGet(0))
    with ExitStack() as resources:
        context = check(driver.cuCtxCreate(0, device))
        resources.callback(driver.cuCtxDestroy, context)
        module = check(driver.cuModuleLoadData(ptx + b"\0"))
        resources.callback(driver.cuModuleUnload, module)
        kernel = check(driver.cuModuleGetFunction(module, request["kernel"].encode()))
        buffers = {}
        arguments = []
        for index, param in enumerate(request.get("params", [])):
            scalar_type = ctypes.c_int64 if param["type"].startswith("i64") else ctypes.c_double
            if param["type"].endswith("[]"):
                host = (scalar_type * len(param["values"]))(*param["values"])
                pointer = check(driver.cuMemAlloc(ctypes.sizeof(host)))
                resources.callback(driver.cuMemFree, pointer)
                check(driver.cuMemcpyHtoD(pointer, ctypes.addressof(host), ctypes.sizeof(host)))
                buffers[index] = (host, pointer)
                arguments.append(ctypes.c_uint64(int(pointer)))
            else:
                arguments.append(scalar_type(param["value"]))
        # Keep argument storage alive until the synchronous launch completes.
        pointers = (ctypes.c_void_p * len(arguments))(*(ctypes.addressof(a) for a in arguments))
        check(
            driver.cuLaunchKernel(
                kernel,
                *request.get("grid", [1, 1, 1]),
                *request.get("block", [1, 1, 1]),
                0,
                0,
                ctypes.addressof(pointers) if arguments else 0,
                0,
            )
        )
        check(driver.cuCtxSynchronize())
        outputs = []
        for output in request.get("outputs", []):
            index = output["param"]
            host, pointer = buffers[index]
            count = output.get("count", len(host))
            if count:
                check(driver.cuMemcpyDtoH(ctypes.addressof(host), pointer, count * 8))
            outputs.append(
                {
                    "param": index,
                    "type": request["params"][index]["type"],
                    "values": list(host)[:count],
                }
            )
        return {"status": "ok", "outputs": outputs}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ptx", type=Path, help="PTX file to execute")
    parser.add_argument("request", type=Path, help="JSON launch options file")
    parser.add_argument("result", type=Path, help="JSON result file to write")
    args = parser.parse_args(argv)
    try:
        response = execute(args.ptx, json.loads(args.request.read_text()))
        # Reject nonfinite GPU results rather than emitting nonstandard JSON.
        text = json.dumps(response, allow_nan=False)
    except Exception as error:  # noqa: BLE001 - all failures belong in the wire response
        text = json.dumps({"status": "error", "error": str(error)})
        status = 1
    else:
        status = 0
    args.result.write_text(text + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
