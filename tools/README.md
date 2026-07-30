# RSMeM MCP Tools

Agent remote-sensing tools: Layer 1 local compute + Layer 2 HTTP service
interfaces.

```
tools/
├── Analysis.py, Index.py, Inversion.py, Statistics.py   # Layer 1
├── Perception.py                                        # Layer 2 (9 backends)
├── service_urls.py
└── requirements-tools.txt
```

> **Release status:** the GPU service deployment package is still being
> prepared and is not included in this release. The files here define the
> public MCP interfaces and expected service endpoints.

---

## Layer 1 — MCP Local Tools

```bash
cd RSMeM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r tools/requirements-tools.txt
```

MCP tool servers are configured in [`configs/agent_mcp.json`](../configs/agent_mcp.json); YAML configs point to it via `agent_config_path`.

---

## Layer 2 — 9 Perception Microservices

`Perception.py` dispatches to nine GPU-backed HTTP services. The corresponding
deployment environments, model weights, launch commands, and health checks
will be released separately after validation. See
[`models/README.md`](../models/README.md) for the checkpoint inventory.
