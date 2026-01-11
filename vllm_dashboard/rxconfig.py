import reflex as rx

config = rx.Config(
    app_name="vllm_dashboard",
    backend_host="0.0.0.0",
    backend_port=8002,
    frontend_port=3000,
)
