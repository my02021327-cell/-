import uvicorn

from backend.app.config import FORWARDED_ALLOW_IPS, HOST, PORT


if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        proxy_headers=True,
        forwarded_allow_ips=FORWARDED_ALLOW_IPS,
    )
