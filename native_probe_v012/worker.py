"""Small native ABI and CUDA/PTX execution gates, not a vLLM benchmark."""
import ctypes as C
import json
import os
from pathlib import Path
import sys
import traceback


# 模块：Linux扩展；作用：实际执行llguidance的原生语法编译；核心业务：否。
def check_llguidance():
    import llguidance as llg
    grammar = llg.LLMatcher.grammar_from_json_schema(json.dumps({"type": "object", "properties": {
        "metric": {"enum": ["gross_margin", "debt_ratio"]}}, "required": ["metric"], "additionalProperties": False}))
    validation = llg.LLMatcher.validate_grammar(grammar)
    if validation:
        raise RuntimeError("Native grammar validation failed: " + validation)
    path = Path(llg.__file__).resolve()
    if not path.is_relative_to(Path(__file__).parent / "payload/site-packages"):
        raise RuntimeError("Loaded llguidance from unexpected environment")
    if llg.__version__ != "1.7.6":
        raise RuntimeError("Unexpected llguidance version")
    return {"version": llg.__version__, "module": str(path), "grammar_compiled": True}


# 模块：GPU兼容性；作用：通过指定驱动库JIT编译PTX并运行4字节写入核；核心业务：否。
def check_cuda(bus_id, library):
    cuda = C.CDLL(library)
    bindings = {
        "cuInit": [C.c_uint], "cuDriverGetVersion": [C.POINTER(C.c_int)],
        "cuDeviceGetByPCIBusId": [C.POINTER(C.c_int), C.c_char_p],
        "cuDeviceGetName": [C.c_char_p, C.c_int, C.c_int],
        "cuCtxCreate_v2": [C.POINTER(C.c_void_p), C.c_uint, C.c_int],
        "cuCtxDestroy_v2": [C.c_void_p], "cuCtxSynchronize": [],
        "cuModuleLoadDataEx": [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint, C.POINTER(C.c_int), C.POINTER(C.c_void_p)],
        "cuModuleGetFunction": [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
        "cuModuleUnload": [C.c_void_p],
        "cuMemAlloc_v2": [C.POINTER(C.c_uint64), C.c_size_t], "cuMemFree_v2": [C.c_uint64],
        "cuMemcpyDtoH_v2": [C.c_void_p, C.c_uint64, C.c_size_t],
        "cuLaunchKernel": [C.c_void_p, *([C.c_uint] * 7), C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)],
        "cuGetErrorString": [C.c_int, C.POINTER(C.c_char_p)],
    }
    for name, args in bindings.items():
        function = getattr(cuda, name)
        function.argtypes, function.restype = args, C.c_int
    def call(name, *args):
        code = getattr(cuda, name)(*args)
        if code:
            message = C.c_char_p()
            cuda.cuGetErrorString(code, C.byref(message))
            raise RuntimeError(f"{name}: CUDA {code}: {(message.value or b'unknown').decode(errors='replace')}")

    call("cuInit", 0)
    version, device = C.c_int(), C.c_int()
    call("cuDriverGetVersion", C.byref(version))
    call("cuDeviceGetByPCIBusId", C.byref(device), bus_id.encode())
    name = C.create_string_buffer(256)
    call("cuDeviceGetName", name, len(name), device)
    context, module, function, pointer = C.c_void_p(), C.c_void_p(), C.c_void_p(), C.c_uint64()
    jit_log = C.create_string_buffer(8192)
    try:
        call("cuCtxCreate_v2", C.byref(context), 0, device)
        ptx = C.create_string_buffer(b'''.version 8.8
.target sm_80
.address_size 64
.visible .entry write_value(.param .u64 output) {
 .reg .b64 %p;
 .reg .b32 %v;
 ld.param.u64 %p, [output];
 mov.u32 %v, 42;
 st.global.u32 [%p], %v;
 ret;
}
''')
        options = (C.c_int * 2)(5, 6)  # CU_JIT_ERROR_LOG_BUFFER / SIZE_BYTES
        values = (C.c_void_p * 2)(C.addressof(jit_log), len(jit_log))
        try:
            call("cuModuleLoadDataEx", C.byref(module), C.cast(ptx, C.c_void_p), 2, options, values)
        except RuntimeError as error:
            raise RuntimeError(str(error) + " JIT=" + jit_log.value.decode(errors="replace")) from error
        call("cuModuleGetFunction", C.byref(function), module, b"write_value")
        call("cuMemAlloc_v2", C.byref(pointer), 4)
        params = (C.c_void_p * 1)(C.addressof(pointer))
        call("cuLaunchKernel", function, 1, 1, 1, 1, 1, 1, 0, None, params, None)
        call("cuCtxSynchronize")
        result = C.c_int()
        call("cuMemcpyDtoH_v2", C.byref(result), pointer, 4)
        if result.value != 42:
            raise RuntimeError("GPU kernel returned incorrect value")
        return {"driver_api_version": version.value, "gpu": name.value.decode(), "pci_bus_id": bus_id,
                "ptx_version": "8.8", "target": "sm_80", "kernel_output": result.value,
                "loaded_libraries": [line.strip() for line in Path("/proc/self/maps").read_text().splitlines()
                                     if "libcuda.so" in line or "libnvidia-ptxjitcompiler" in line]}
    finally:
        if pointer.value:
            cuda.cuMemFree_v2(pointer)
        if module.value:
            cuda.cuModuleUnload(module)
        if context.value:
            cuda.cuCtxDestroy_v2(context)


if __name__ == "__main__":
    try:
        if sys.argv[1] == "llguidance":
            data = check_llguidance()
        else:
            data = check_cuda(sys.argv[2], sys.argv[3])
        print("PROBE_RESULT=" + json.dumps({"passed": True, **data}), flush=True)
    except Exception as error:
        traceback.print_exc()
        print("PROBE_RESULT=" + json.dumps({"passed": False, "error": str(error)}), flush=True)
        raise SystemExit(1)
