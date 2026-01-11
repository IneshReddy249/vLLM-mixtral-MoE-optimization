import reflex as rx
from .state import State


def metric_box(label: str, value, unit: str, color: str):
    return rx.box(
        rx.vstack(
            rx.text(label, font_size="11px", color="#888"),
            rx.hstack(
                rx.text(value, font_size="24px", font_weight="bold", color=color),
                rx.text(unit, font_size="11px", color="#666"),
                spacing="1",
                align_items="baseline",
            ),
            spacing="0",
            align_items="center",
        ),
        padding="12px 20px",
        background="#111",
        border_radius="12px",
        border=f"1px solid {color}22",
        min_width="100px",
    )


def message_bubble(msg: dict):
    is_user = msg["role"] == "user"
    return rx.box(
        rx.box(
            rx.text(
                msg["content"],
                color=rx.cond(is_user, "#fff", "#e5e5e5"),
                font_size="14px",
                white_space="pre-wrap",
                line_height="1.6",
            ),
            padding="10px 16px",
            background=rx.cond(is_user, "#2a2a2a", "#111"),
            border_radius=rx.cond(is_user, "18px 18px 4px 18px", "18px 18px 18px 4px"),
            display="inline-block",
            max_width="85%",
        ),
        text_align=rx.cond(is_user, "right", "left"),
        width="100%",
        margin_bottom="12px",
    )


def index():
    return rx.box(
        # STICKY HEADER - Title + Metrics
        rx.box(
            rx.box(
                rx.hstack(
                    rx.box(width="40px"),
                    rx.text(
                        "Mixtral 8x7B MoE • vLLM Optimized • 2×A100",
                        font_size="20px",
                        font_weight="bold",
                        color="white",
                        text_align="center",
                        flex="1",
                    ),
                    rx.button(
                        "🗑",
                        on_click=State.clear_chat,
                        background="transparent",
                        color="#666",
                        font_size="18px",
                        padding="8px",
                        border_radius="8px",
                        cursor="pointer",
                        _hover={"color": "#fff", "background": "#333"},
                        width="40px",
                    ),
                    width="100%",
                    justify="between",
                    align="center",
                ),
                padding_top="16px",
                padding_bottom="12px",
            ),
            rx.box(
                rx.hstack(
                    metric_box("TTFT", State.last_ttft.to(int), "ms", "#10b981"),
                    metric_box("Speed", State.last_tps.to(int), "tok/s", "#f59e0b"),
                    metric_box("ITL", State.itl_avg.to(int), "ms", "#06b6d4"),
                    metric_box("Tokens", State.last_tokens, "", "#a855f7"),
                    metric_box("Latency", (State.last_latency / 1000), "s", "#3b82f6"),
                    spacing="3",
                    justify="center",
                ),
                padding_bottom="16px",
            ),
            position="sticky",
            top="0",
            z_index="100",
            background="#0a0a0a",
            border_bottom="1px solid #222",
            padding_x="20px",
        ),
        # CHAT MESSAGES - Scrollable area
        rx.box(
            rx.box(
                rx.foreach(State.messages, message_bubble),
                max_width="800px",
                margin_left="auto",
                margin_right="auto",
                padding_x="20px",
                padding_top="20px",
                padding_bottom="140px",
            ),
        ),
        # INPUT AREA - Fixed at bottom
        rx.box(
            rx.box(
                rx.hstack(
                    rx.text_area(
                        value=State.input_text,
                        on_change=State.set_input_text,
                        placeholder="Message Mixtral...",
                        width="100%",
                        min_height="56px",
                        max_height="200px",
                        background="#111",
                        border="1px solid #333",
                        border_radius="16px",
                        color="white",
                        font_size="15px",
                        padding="16px",
                        padding_right="60px",
                        resize="none",
                    ),
                    rx.button(
                        rx.cond(State.is_generating, "●", "↑"),
                        on_click=State.generate,
                        disabled=State.is_generating,
                        position="absolute",
                        right="12px",
                        bottom="12px",
                        background=rx.cond(State.is_generating, "#444", "#10b981"),
                        color="white",
                        border_radius="8px",
                        width="36px",
                        height="36px",
                        padding="0",
                        font_size="18px",
                        font_weight="bold",
                    ),
                    position="relative",
                    width="100%",
                ),
                max_width="700px",
                margin_left="auto",
                margin_right="auto",
                padding="8px",
            ),
            position="fixed",
            bottom="20px",
            left="0",
            right="0",
            padding_x="20px",
            background="linear-gradient(transparent, #0a0a0a 20%)",
            padding_top="40px",
        ),
        background="#0a0a0a",
        min_height="100vh",
        width="100%",
    )


app = rx.App(theme=rx.theme(appearance="dark"))
app.add_page(index)
