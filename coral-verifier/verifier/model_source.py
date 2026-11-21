"""Optional model from a URL instead of the built-in one (add-on options model_url, model_sha256, model_token).

Empty model_url = the built-in model baked into the image (the default; nothing is downloaded). Otherwise the file is downloaded once into the
add-on's own storage (/data/models/<sha256>.tflite) and used only if its sha256 is exactly model_sha256 and it is a TFLite flatbuffer; a later
start reuses the stored copy (no network). Any problem -> the built-in model, with the reason in the log and in /v1/health.
A model the Coral could not run (the verifier gave up recovering it, see server.main) is marked <sha256>.tflite.failed and is not used again:
changing model_sha256 (another file) or deleting the marker clears it.
model_token is sent as a bearer token (a GitHub fine-grained token with Contents: read for a private repository). For an api.github.com URL
(.../repos/<owner>/<repo>/contents/<path>?ref=<commit>) the raw file is requested. The token is never logged.
"""
import hashlib, logging, os, re, urllib.parse, urllib.request

log = logging.getLogger("verifier.model")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_BYTES = 64 * 1024 * 1024            # an Edge TPU model of this family is ~7 MB
TIMEOUT_S = 60.0
STORE = "/data/models"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, token, dst, timeout=TIMEOUT_S, max_bytes=MAX_BYTES):
    """Downloads url to dst (https only, size-capped). Raises on any failure."""
    headers = {"User-Agent": "coral-verifier"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if urllib.parse.urlparse(url).hostname == "api.github.com":
        headers["Accept"] = "application/vnd.github.raw"           # the file itself, not its JSON description
    req = urllib.request.Request(url, headers=headers)
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            n += len(b)
            if n > max_bytes:
                raise ValueError(f"download larger than {max_bytes // 2**20} MB")
            f.write(b)


def check_file(path):
    with open(path, "rb") as f:
        head = f.read(8)
    if len(head) < 8 or head[4:8] != b"TFL3":
        raise ValueError("the file is not a TFLite model")


def failed_marker(store, sha):
    return os.path.join(store, sha + ".tflite.failed")


def resolve(opts, builtin, store=STORE, fetcher=fetch):
    """builtin: the image's model spec ({"kind", "path", "size"}). Returns (spec, origin); origin describes the model for /v1/health."""
    url = str(opts.get("model_url") or "").strip()
    if not url:
        return builtin, {"source": "built-in"}
    sha = str(opts.get("model_sha256") or "").strip().lower()
    name = os.path.basename(urllib.parse.urlparse(url).path) or "model"
    try:
        if urllib.parse.urlparse(url).scheme != "https":
            raise ValueError("model_url must start with https://")
        if not SHA_RE.match(sha):
            raise ValueError("model_sha256 must be the file's sha256 (64 hex characters)")
        path = os.path.join(store, sha + ".tflite")
        if os.path.exists(failed_marker(store, sha)):
            raise ValueError(f"this model made the Coral fail before ({os.path.basename(failed_marker(store, sha))}); "
                             "set another model or delete the marker to retry")
        if not (os.path.exists(path) and sha256_file(path) == sha):
            os.makedirs(store, exist_ok=True)
            part = path + ".part"
            log.info("downloading model %s", name)
            try:
                fetcher(url, str(opts.get("model_token") or ""), part)
                got = sha256_file(part)
                if got != sha:
                    raise ValueError(f"downloaded file has sha256 {got[:12]}..., model_sha256 is {sha[:12]}...")
                check_file(part)
                os.replace(part, path)
            finally:
                if os.path.exists(part):
                    os.remove(part)
            for f in os.listdir(store):                               # keep only the model in use (and failure markers)
                if f.endswith(".tflite") and f != os.path.basename(path):
                    os.remove(os.path.join(store, f))
        check_file(path)
    except Exception as e:
        reason = str(e) or type(e).__name__
        log.error("model_url not used, running the built-in model: %s", reason)
        return builtin, {"source": "built-in", "fallback": reason}
    log.info("model %s (sha256 %s...) from model_url", name, sha[:12])
    return {**builtin, "path": path, "size": None}, {"source": "url", "file": name, "sha256": sha}   # size None: the model's own input size


def mark_failed(origin, store=STORE):
    """The Coral could not run the model from model_url: do not use it on the next start."""
    if origin.get("source") != "url":
        return
    try:
        with open(failed_marker(store, origin["sha256"]), "w") as f:
            f.write("the verifier gave up recovering the Coral with this model\n")
        log.error("model %s marked as failed: the next start uses the built-in model", origin.get("file"))
    except OSError as e:
        log.error("could not mark model %s as failed: %s", origin.get("file"), e)
