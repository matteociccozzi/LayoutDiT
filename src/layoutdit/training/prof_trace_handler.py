import os

import fsspec
from torch.profiler import profiler

_fs = fsspec.filesystem("gcs")

_local_trace_dir = "/tmp/layoutdit_profiler"
os.makedirs(_local_trace_dir, exist_ok=True)


def trace_handler(prof: profiler.profile, run_name: str):
    """
    Profiler trace handler that copies from local fs to gs
    """
    trace_name = f"trace_{prof.step_num}.json"
    local_path = os.path.join(_local_trace_dir, trace_name)
    prof.export_chrome_trace(local_path)

    gs_path = f"gs://layoutdit/{run_name}/profiler/{trace_name}"
    with _fs.open(gs_path, "wb") as f:
        with open(local_path, "rb") as lf:
            f.write(lf.read())
