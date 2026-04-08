# PhoGPT Deployment

PhoGPT khong nen chay chung env voi backend chinh. Model nay can runtime rieng.

## 1. Build service

```bash
docker compose -f docker-compose.phogpt.yml build
```

## 2. Run service

```bash
docker compose -f docker-compose.phogpt.yml up
```

Service se mo o:

```bash
http://127.0.0.1:8001
```

Neu can thu mot snapshot khac cua PhoGPT:

```bash
export PHOGPT_REVISION="<commit-or-tag>"
export PHOGPT_FORCE_DOWNLOAD="true"
```

Hoac set truc tiep trong `docker-compose.phogpt.yml`.

## 3. Kiem tra health

```bash
curl http://127.0.0.1:8001/health
```

Ban chi nen noi backend chinh vao PhoGPT khi `loaded` la `true`.

## 4. Noi backend chinh vao PhoGPT service

Trong shell chay backend chinh:

```bash
export PHOGPT_BASE_URL="http://127.0.0.1:8001"
```

Neu service co auth:

```bash
export PHOGPT_AUTH_HEADER="Bearer <token>"
```

Sau do restart backend chinh.

## 5. Kien truc

- backend chinh:
  - Qwen / Vistral / VinaLLaMA qua `HFAdapter`
  - PhoGPT qua `APIAdapter`
- PhoGPT service:
  - `backend/phogpt_service.py`

## 6. Luu y

- Dockerfile nay moi tao khung trien khai rieng, khong dam bao PhoGPT se chay duoc neu model van doi stack Triton cu.
- Neu `/health` van bao `loaded=false`, ban can dung image/env khac tu repo hoặc huong dan chinh thuc cua PhoGPT.
