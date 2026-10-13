"""Performance metrics for GET /v1/metrics: 60 s rolling counters (one bucket per second in a ring, O(1) per event, nothing logged per frame) and a
lightweight 5 s background sampler for host/process CPU, load, memory and Intel GPU busy share. Every reading degrades to None, never raises.
The paths are module-level constants so tests can point them at fake files."""
import os, threading, time

PROC_STAT = "/proc/stat"
PROC_LOADAVG = "/proc/loadavg"
PROC_MEMINFO = "/proc/meminfo"
PROC_SELF_STAT = "/proc/self/stat"
PROC_SELF_STATM = "/proc/self/statm"
PROC_PID_STAT = "/proc/{pid}/stat"
DRM_RC6_GLOB = "/sys/class/drm/card*/gt/gt0/rc6_residency_ms"
SAMPLE_EVERY_S = 5.0
WINDOW_S = 60
ROLL_NAMES = ("busy_ms", "inv", "frames", "stale", "r429", "sl", "sl_confirmed", "sl_lowered", "sl_rejected", "sl_reused")
try:
    CLK_TCK = os.sysconf("SC_CLK_TCK")
except (ValueError, OSError, AttributeError):
    CLK_TCK = 100


class Rolling:
    """Per-second bucket ring over the last WINDOW_S seconds, several named counters."""

    def __init__(self, names, clock=time.monotonic):
        self.names, self.clock = tuple(names), clock
        self.lock = threading.Lock()
        self.sec = [None] * (WINDOW_S + 1)
        self.val = [dict.fromkeys(self.names, 0.0) for _ in range(WINDOW_S + 1)]

    def add(self, name, n=1.0):
        s = int(self.clock())
        i = s % (WINDOW_S + 1)
        with self.lock:
            if self.sec[i] != s:
                self.sec[i] = s
                self.val[i] = dict.fromkeys(self.names, 0.0)
            self.val[i][name] += n

    def sum(self, name, window=WINDOW_S):
        now = int(self.clock())
        with self.lock:
            return sum(self.val[i][name] for i, s in enumerate(self.sec) if s is not None and 0 <= now - s < window)


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except (OSError, ValueError):
        return None


def _cpu_times(text):
    """/proc/stat first line -> (busy, total) jiffies, or None."""
    try:
        f = [float(x) for x in text.splitlines()[0].split()[1:]]
        total = sum(f[:8])
        idle = f[3] + (f[4] if len(f) > 4 else 0.0)
        return total - idle, total
    except (AttributeError, IndexError, ValueError):
        return None


def _stat_cpu_s(text):
    """/proc/<pid>/stat -> utime+stime seconds (fields after the last ')')."""
    try:
        f = text[text.rindex(")") + 2:].split()
        return (int(f[11]) + int(f[12])) / CLK_TCK
    except (AttributeError, ValueError, IndexError):
        return None


def _rc6_ms():
    import glob
    try:
        for p in sorted(glob.glob(DRM_RC6_GLOB)):
            t = _read(p)
            if t is not None:
                return float(t.strip())
    except (OSError, ValueError):
        pass
    return None


class Sampler:
    """Samples every SAMPLE_EVERY_S in its own daemon thread; requests only read the last computed values."""

    def __init__(self, child_pid_fn=lambda: None, clock=time.monotonic):
        self.child_pid_fn, self.clock = child_pid_fn, clock
        self.lock = threading.Lock()
        self.prev = None
        self.v = {"host_cpu": None, "gpu_busy": None, "gpu_ok": False, "proc_cpu": None}
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._run, name="metrics-sampler", daemon=True).start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(SAMPLE_EVERY_S)

    def _proc_cpu_s(self):
        own = _stat_cpu_s(_read(PROC_SELF_STAT))
        t = os.times()
        if own is None:
            own = t.user + t.system
        total = own + t.children_user + t.children_system        # reaped children
        try:
            pid = self.child_pid_fn()
        except Exception:
            pid = None
        if pid:
            c = _stat_cpu_s(_read(PROC_PID_STAT.format(pid=pid)))      # the live Coral child
            if c is not None:
                total += c
        return total

    def sample(self):
        try:
            now = self.clock()
            host, rc6, pcpu = _cpu_times(_read(PROC_STAT)), _rc6_ms(), self._proc_cpu_s()
            out = dict(self.v)
            out["gpu_ok"] = rc6 is not None
            if self.prev is not None:
                pnow, phost, prc6, ppcpu = self.prev
                dt = now - pnow
                if host and phost and host[1] > phost[1]:
                    out["host_cpu"] = round(100.0 * (host[0] - phost[0]) / (host[1] - phost[1]), 1)
                if rc6 is not None and prc6 is not None and dt > 0:
                    out["gpu_busy"] = round(min(100.0, max(0.0, 100.0 - 100.0 * (rc6 - prc6) / (dt * 1000.0))), 1)
                elif rc6 is None:
                    out["gpu_busy"] = None
                if dt > 0 and pcpu is not None and ppcpu is not None:
                    out["proc_cpu"] = round(max(0.0, 100.0 * (pcpu - ppcpu) / dt), 1)
            self.prev = (now, host, rc6, pcpu)
            with self.lock:
                self.v = out
        except Exception:
            pass

    def values(self):
        with self.lock:
            return dict(self.v)


def host_instant():
    """load1 and used-memory share, read per request (two tiny files)."""
    load1 = mem = None
    try:
        load1 = round(float(_read(PROC_LOADAVG).split()[0]), 2)
    except (AttributeError, IndexError, ValueError):
        pass
    try:
        m = {}
        for line in _read(PROC_MEMINFO).splitlines():
            k, _, v = line.partition(":")
            m[k] = float(v.split()[0])
        mem = round(100.0 * (1.0 - m["MemAvailable"] / m["MemTotal"]), 1)
    except (AttributeError, KeyError, IndexError, ValueError, ZeroDivisionError):
        pass
    return load1, mem


def rss_mb():
    try:
        return round(int(_read(PROC_SELF_STATM).split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1048576.0, 1)
    except (AttributeError, IndexError, ValueError, OSError):
        return None
