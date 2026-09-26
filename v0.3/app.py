import gradio as gr
from dotenv import load_dotenv
import re
import json
from datetime import datetime, timezone

from implementation.answer import answer_question
from implementation.extractor import extract_metadata, build_transcript
from implementation.database import (
    init_db,
    upsert_user,
    insert_ticket,
    insert_chat_session,
)

load_dotenv(override=True)

# Initialise the database tables on startup
init_db()

# 1. Update CSS to force 'start' alignment for chatbot Markdown elements
custom_css = """
#bidi-input textarea {
    unicode-bidi: plaintext;
    text-align: start;
}

#bidi-chatbot .prose p, 
#bidi-chatbot .prose li, 
#bidi-chatbot .prose ul, 
#bidi-chatbot .prose ol {
    text-align: start !important;
}
"""

def strip_bidi_tags(text):
    """Removes the HTML wrapper before sending history to the backend"""
    text = text.replace('<div dir="auto">\n\n', '')
    text = text.replace('\n\n</div>', '')
    return text


def chat(history):
    # Clean the current message so the AI doesn't see HTML
    last_msg_content = history[-1]["content"]
    if isinstance(last_msg_content, list):
        last_msg_str = "".join([b.get("text", "") for b in last_msg_content if b.get("type") == "text"])
    else:
        last_msg_str = str(last_msg_content)
    last_message = strip_bidi_tags(last_msg_str)
    
    # Clean the prior history context
    prior = []
    for msg in history[:-1]:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            msg_str = "".join([b.get("text", "") for b in msg_content if b.get("type") == "text"])
        else:
            msg_str = str(msg_content)
        prior.append({"role": msg["role"], "content": strip_bidi_tags(msg_str)})
        
    answer, context = answer_question(last_message, prior)
    
    # Automatically add a space after a number and period if it is missing (1.افتح -> 1. افتح)
    answer = re.sub(r'(?m)^(\s*\d+)\.(?=[^\s])', r'\1. ', answer)
    
    # Wrap the AI's answer in the BiDi div to natively fix punctuation and list markers
    formatted_answer = f'<div dir="auto">\n\n{answer}\n\n</div>'
    
    history.append(gr.ChatMessage(role="assistant", content=formatted_answer))
    return history


# ---------------------------------------------------------------------------
# Session closure pipeline (Section 3 + 4 + 5 of the v0.3 spec)
# ---------------------------------------------------------------------------

def _clean_history_for_extraction(history: list[dict]) -> list[dict]:
    """Strip BiDi HTML wrappers from every message for the LLM extractor."""
    cleaned = []
    for msg in history:
        content = msg["content"]
        if isinstance(content, list):
            text = "".join(
                b.get("text", "") for b in content if b.get("type") == "text"
            )
        else:
            text = str(content)
        cleaned.append({"role": msg["role"], "content": strip_bidi_tags(text)})
    return cleaned


def end_session(history, session_start):
    """
    Triggered by the 'End Session' button.
    1. Disables the text input (state freeze).
    2. Runs the LLM extraction pipeline.
    3. Executes INSERT / UPSERT operations across the three DB tables.
    4. Returns a closure message to the chatbot.
    """
    if not history:
        history.append(
            gr.ChatMessage(
                role="assistant",
                content='<div dir="auto">\n\nلا توجد محادثة لإنهائها.\n\n</div>',
            )
        )
        return (
            history,
            gr.update(interactive=False),   # disable textbox
            gr.update(interactive=False),   # disable end-session button
            session_start,
        )

    end_time = datetime.now(timezone.utc)
    start_time = session_start or end_time

    # --- 1. Extract metadata via LLM ----------------------------------
    cleaned = _clean_history_for_extraction(history)
    metadata = extract_metadata(cleaned)

    # --- 2. UPSERT user ------------------------------------------------
    extracted_user_id = metadata.get("user_id") or "anonymous"
    upsert_user(
        user_id=extracted_user_id,
        college=metadata.get("college"),
        role=metadata.get("role"),
    )

    # --- 3. INSERT ticket -----------------------------------------------
    ticket = insert_ticket(
        user_id=extracted_user_id,
        ticket_class=metadata.get("ticket_class"),
        sub_class=metadata.get("sub_class"),
        severity=metadata.get("severity"),
        external_ticket_ref=metadata.get("external_ticket_ref"),
    )

    # --- 4. INSERT chat_session -----------------------------------------
    raw_transcript = json.dumps(cleaned, ensure_ascii=False)
    insert_chat_session(
        ticket_id=ticket.ticket_id,
        start_time=start_time,
        end_time=end_time,
        ticket_summary=metadata.get("session_summary"),
        issue_resolved=metadata.get("issue_resolved", False),
        sentiment=metadata.get("sentiment"),
        raw_transcript=raw_transcript,
    )

    # --- 5. Closure message ---------------------------------------------
    closure = (
        "✅ تم إنهاء الجلسة بنجاح وحفظ البيانات.\n\n"
        f"**رقم التذكرة:** `{ticket.ticket_id}`"
    )
    history.append(
        gr.ChatMessage(
            role="assistant",
            content=f'<div dir="auto">\n\n{closure}\n\n</div>',
        )
    )

    return (
        history,
        gr.update(interactive=False),   # disable textbox
        gr.update(interactive=False),   # disable end-session button
        session_start,
    )


def main():
    def put_message_in_chatbot(message, history):
        formatted_message = f'<div dir="auto">\n\n{message}\n\n</div>'
        history.append(gr.ChatMessage(role="user", content=formatted_message))
        return "", history

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    # تمرير theme و css هنا في Blocks مثل v0.2
    with gr.Blocks(title="خبير نظم جامعة الملك سعود", theme=theme, css=custom_css) as ui:

        # --- Data Privacy Banner (Section 2) ---
        gr.Markdown(
            "> **تنبيه الخصوصية:** يُرجى العلم بأنه يتم تسجيل وتحليل هذه المحادثة "
            "لأغراض ضمان الجودة وتطوير الخدمات وفقاً لسياسات حماية البيانات."
        )

        gr.Markdown("# خبير نظم جامعة الملك سعود")

        # Hidden state to track session start time
        session_start = gr.State(value=datetime.now(timezone.utc))

        with gr.Row():
            with gr.Column(scale=1):
                # استخدام show_copy_button بدلاً من buttons
                chatbot = gr.Chatbot(
                    label="", 
                    height=600, 
                    show_copy_button=True,
                    type="messages",
                    elem_id="bidi-chatbot"
                )
                
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="اسألني عن أي مشكلة تقنية في نظم جامعة الملك سعود وسأحاول مساعدتك",
                    show_label=False,
                    elem_id="bidi-input" 
                )

                # --- End Session button (Section 2) ---
                end_session_btn = gr.Button(
                    "إنهاء الجلسة",
                    variant="stop",
                    elem_id="end-session-btn",
                )

        # --- Event wiring ---
        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot])

        # End Session pipeline (Section 5)
        end_session_btn.click(
            end_session,
            inputs=[chatbot, session_start],
            outputs=[chatbot, message, end_session_btn, session_start],
        )

    # تشغيل صافي بدون تمرير theme أو css هنا
    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()