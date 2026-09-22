import os

# 必须在 import huggingface_hub 之前设置这些环境变量才生效。
# 关闭 Xet 分块存储和 hf_transfer 高性能模式：这台机器连 Xet/CDN 容易在传输大 zip
# 时挂起（TCP 连上但数据流卡死），退回传统 HTTP 分块下载更稳、断了能续传。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
# 走内部代理（与其它下载一致）；已设置则不覆盖。
os.environ.setdefault("https_proxy", "http://agent.baidu.com:8188")
os.environ.setdefault("http_proxy", "http://agent.baidu.com:8188")

from huggingface_hub import snapshot_download

# HF_TOKEN 从环境变量读取，避免匿名 IP 被限流（之前遇到过 429）。
# 运行前先 export HF_TOKEN=hf_xxx（无 token 也能跑，但更容易被限流）。
snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    allow_patterns=["background_texture.zip", "embodiments.zip", "objects.zip"],
    local_dir=".",
    repo_type="dataset",
    resume_download=True,
    token=os.environ.get("HF_TOKEN"),
    max_workers=1,          # 串行下载，降低并发对不稳定连接的压力
    etag_timeout=30,        # HEAD/元数据请求超时放宽到 30s
)

