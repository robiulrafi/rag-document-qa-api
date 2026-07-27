# ---- Base image ----
# Python 3.12 (matches your venv312). "slim" = smaller image, no build bloat.
# NOT 3.14 — that's the interpreter that breaks your ML wheels.
FROM python:3.12-slim

# ---- Working directory ----
# All following commands run from /app inside the container.
WORKDIR /app

# ---- System deps (only what's needed) ----
# Some Python packages need a C compiler / build tools to install wheels.
# We install, use, and clean up in one layer to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps FIRST (caching optimization) ----
# Copy ONLY requirements.txt before the app code. Docker caches each layer;
# as long as requirements.txt doesn't change, Docker reuses the cached
# install layer on rebuilds — so you don't re-download torch every time you
# edit a .py file. Copy code first and this cache breaks on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- App code ----
# Now copy the rest of the project (respects .dockerignore).
COPY . .

# ---- Port ----
# Document that the app listens on 8000. (EXPOSE is metadata; the actual
# publish happens with -p at run time.)
EXPOSE 8000

# ---- Start command ----
# Run uvicorn, binding to 0.0.0.0 so it's reachable from OUTSIDE the container
# (127.0.0.1 would only be reachable inside it). Points at your real app object.
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
