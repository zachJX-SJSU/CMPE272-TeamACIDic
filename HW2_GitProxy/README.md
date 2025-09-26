# GitHub API proxy server


## Features
- Issue CRUD & comments
- Webhook receiver with HMAC validation (issues / issue_comment / ping)
- OpenAPI 3.1 spec (`openapi.yaml`)
- Tests with `pytest` + `respx`
- Docker & devcontainer


## Configure
See `.env.sample`. Use a **fine-grained PAT** scoped to your repo (Issues: Read/Write).


## Run
```bash
docker build -t issues-gw .
docker run --env-file .env -p ${PORT}:${PORT} issues-gw