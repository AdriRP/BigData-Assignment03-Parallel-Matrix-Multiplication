"""Record the hardware and software environment to results/environment.json."""
import json
import platform
import subprocess
import sys

import numpy
import psutil

from common import CONFIG, RESULTS


def cpu_name():
    if sys.platform == "win32":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def tool_output(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        return (p.stdout + p.stderr).strip()
    except OSError as e:
        return f"unavailable: {e}"


def windows_cpu_details():
    """Cache sizes and core counts as reported by Windows (Win32_Processor)."""
    if sys.platform != "win32":
        return "not Windows"
    ps = ("Get-CimInstance Win32_Processor | Select-Object NumberOfCores,NumberOfLogicalProcessors,"
          "L2CacheSize,L3CacheSize,MaxClockSpeed | ConvertTo-Json")
    try:
        return json.loads(tool_output(["powershell", "-NoProfile", "-Command", ps]))
    except json.JSONDecodeError:
        return "unavailable"


def windows_topology():
    """Logical processors of each physical core and of each L3 cache (GetLogicalProcessorInformationEx)."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    def query(rel):
        size = wintypes.DWORD(0)
        k32.GetLogicalProcessorInformationEx(rel, None, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        if not k32.GetLogicalProcessorInformationEx(rel, buf, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        out, off = [], 0
        while off < size.value:
            length = int.from_bytes(buf.raw[off + 4:off + 8], "little")
            out.append(buf.raw[off + 8:off + length])
            off += length
        return out

    def bits(mask):
        return [i for i in range(64) if mask >> i & 1]

    # PROCESSOR_RELATIONSHIP: the GROUP_AFFINITY mask starts at byte 24 of the body
    cores = [bits(int.from_bytes(b[24:32], "little")) for b in query(0)]
    # CACHE_RELATIONSHIP: Level at byte 0, CacheSize at 4, GROUP_AFFINITY mask at 32
    l3 = [{"size_mb": int.from_bytes(b[4:8], "little") // 2 ** 20, "logical_processors": bits(int.from_bytes(b[32:40], "little"))}
          for b in query(2) if b[0] == 3]
    return {"cores": cores, "l3_caches": l3}


def main():
    RESULTS.mkdir(exist_ok=True)
    env = {
        "cpu": cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_processors": psutil.cpu_count(logical=True),
        "win32_processor (L2/L3 in KB, clock in MHz)": windows_cpu_details(),
        "topology": windows_topology() if sys.platform == "win32" else "not Windows",
        "ram_bytes": psutil.virtual_memory().total,
        "os": platform.platform(),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "numpy (inputs/validation only)": numpy.__version__,
        "psutil (memory measurement)": psutil.__version__,
        "java": tool_output(["java", "-version"]),
        "javac": tool_output(["javac", "-version"]),
        "jvm_flags": " ".join(CONFIG["jvm_flags"]),
        "jvm_flags_no_vectorisation": " ".join(CONFIG["jvm_flags_no_vectorisation"]),
        "jvm_ergonomic_flags": tool_output(["java", *CONFIG["jvm_flags"], "-XX:+PrintCommandLineFlags", "-version"]).splitlines()[0],
        "jvm_UseSuperWord_default": [l.strip() for l in tool_output(["java", "-XX:+PrintFlagsFinal", "-version"]).splitlines() if "UseSuperWord" in l],
        "affinity": ("placement 'pinned': the driver restricts each process (psutil cpu_affinity) to the logical "
                     "processors given by common.pinned_cpus(p); placement 'default': no restriction, the Windows "
                     "scheduler places the threads. Normal priority in both."),
        "kernel_threads": "seq: 1 (main thread); par: p pool workers, the main thread waits. No numerical library is used.",
    }
    with open(RESULTS / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
