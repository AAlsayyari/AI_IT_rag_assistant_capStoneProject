# import gradio as gr
# from dotenv import load_dotenv
# import re
# import json
# from datetime import datetime, timezone

# from implementation.answer import answer_question
# from implementation.extractor import extract_metadata, build_transcript
# from implementation.database import (
#     init_db,
#     upsert_user,
#     insert_ticket,
#     insert_chat_session,
# )

# load_dotenv(override=True)

# # Initialise the database tables on startup
# init_db()

# # 1. Update CSS to force 'start' alignment for chatbot Markdown elements
# custom_css = """
# #bidi-input textarea {
#     unicode-bidi: plaintext;
#     text-align: start;
# }

# #bidi-chatbot .prose p, 
# #bidi-chatbot .prose li, 
# #bidi-chatbot .prose ul, 
# #bidi-chatbot .prose ol {
#     text-align: start !important;
# }
# """

# def strip_bidi_tags(text):
#     """Removes the HTML wrapper before sending history to the backend"""
#     text = text.replace('<div dir="auto">\n\n', '')
#     text = text.replace('\n\n</div>', '')
#     return text


# # without returning the context.

# # def chat(history):
# #     # Clean the current message so the AI doesn't see HTML
# #     last_msg_content = history[-1]["content"]
# #     if isinstance(last_msg_content, list):
# #         last_msg_str = "".join([b.get("text", "") for b in last_msg_content if b.get("type") == "text"])
# #     else:
# #         last_msg_str = str(last_msg_content)
# #     last_message = strip_bidi_tags(last_msg_str)
    
# #     # Clean the prior history context
# #     prior = []
# #     for msg in history[:-1]:
# #         msg_content = msg["content"]
# #         if isinstance(msg_content, list):
# #             msg_str = "".join([b.get("text", "") for b in msg_content if b.get("type") == "text"])
# #         else:
# #             msg_str = str(msg_content)
# #         prior.append({"role": msg["role"], "content": strip_bidi_tags(msg_str)})
        
# #     answer, context = answer_question(last_message, prior)
    
# #     # Automatically add a space after a number and period if it is missing (1.افتح -> 1. افتح)
# #     answer = re.sub(r'(?m)^(\s*\d+)\.(?=[^\s])', r'\1. ', answer)
    
# #     # Wrap the AI's answer in the BiDi div to natively fix punctuation and list markers
# #     formatted_answer = f'<div dir="auto">\n\n{answer}\n\n</div>'
    
# #     history.append(gr.ChatMessage(role="assistant", content=formatted_answer))
# #     return history





# # to return the context.
# def chat(history):
#     # Clean the current message so the AI doesn't see HTML
#     last_msg_content = history[-1]["content"]
#     if isinstance(last_msg_content, list):
#         last_msg_str = "".join([b.get("text", "") for b in last_msg_content if b.get("type") == "text"])
#     else:
#         last_msg_str = str(last_msg_content)
#     last_message = strip_bidi_tags(last_msg_str)
    
#     # Clean the prior history context
#     prior = []
#     for msg in history[:-1]:
#         msg_content = msg["content"]
#         if isinstance(msg_content, list):
#             msg_str = "".join([b.get("text", "") for b in msg_content if b.get("type") == "text"])
#         else:
#             msg_str = str(msg_content)
#         prior.append({"role": msg["role"], "content": strip_bidi_tags(msg_str)})
        
#     # 1. الحصول على الإجابة والـ Context المسترجع من Vector DB
#     answer, context = answer_question(last_message, prior)
    
#     # Automatically add a space after a number and period if it is missing
#     answer = re.sub(r'(?m)^(\s*\d+)\.(?=[^\s])', r'\1. ', answer)
    
#     # 2. تنسيق الـ Context المسترجع لعرضه بشكل شفاف
#     formatted_context = ""
#     if context:
#         # إذا كان الـ context عبارة عن قائمة من الـ Chunks/Documents
#         if isinstance(context, list):
#             chunks_text = "\n\n---\n\n".join([f"**Chunk {i+1}:**\n{doc}" for i, doc in enumerate(context)])
#         else:
#             chunks_text = str(context)
            
#         formatted_context = f"""
# <details>
# <summary>🔍 <b>عرض النصوص المسترجعة من قاعدة البيانات (Retrieved Chunks)</b></summary>

# {chunks_text}

# </details>
# """
#     else:
#         formatted_context = """
# <details>
# <summary>⚠️ <b>لم يتم استرجاع أي سياق من قاعدة البيانات (No Context Found)</b></summary>

# النموذج يجيب بناءً على التعليمات العامة أو لم يجد مستنداً مطابقاً.

# </details>
# """

#     # 3. دمج الإجابة مع قسم الـ Chunks
#     full_response = f"{answer}\n\n{formatted_context}"
    
#     # Wrap in BiDi div
#     formatted_answer = f'<div dir="auto">\n\n{full_response}\n\n</div>'
    
#     history.append(gr.ChatMessage(role="assistant", content=formatted_answer))
#     return history



# # ---------------------------------------------------------------------------
# # Session closure pipeline (Section 3 + 4 + 5 of the v0.3 spec)
# # ---------------------------------------------------------------------------

# def _clean_history_for_extraction(history: list[dict]) -> list[dict]:
#     """Strip BiDi HTML wrappers from every message for the LLM extractor."""
#     cleaned = []
#     for msg in history:
#         content = msg["content"]
#         if isinstance(content, list):
#             text = "".join(
#                 b.get("text", "") for b in content if b.get("type") == "text"
#             )
#         else:
#             text = str(content)
#         cleaned.append({"role": msg["role"], "content": strip_bidi_tags(text)})
#     return cleaned


# def end_session(history, session_start):
#     """
#     Triggered by the 'End Session' button.
#     1. Disables the text input (state freeze).
#     2. Runs the LLM extraction pipeline.
#     3. Executes INSERT / UPSERT operations across the three DB tables.
#     4. Returns a closure message to the chatbot.
#     """
#     if not history:
#         history.append(
#             gr.ChatMessage(
#                 role="assistant",
#                 content='<div dir="auto">\n\nلا توجد محادثة لإنهائها.\n\n</div>',
#             )
#         )
#         return (
#             history,
#             gr.update(interactive=False),   # disable textbox
#             gr.update(interactive=False),   # disable end-session button
#             session_start,
#         )

#     end_time = datetime.now(timezone.utc)
#     start_time = session_start or end_time

#     # --- 1. Extract metadata via LLM ----------------------------------
#     cleaned = _clean_history_for_extraction(history)
#     metadata = extract_metadata(cleaned)

#     # --- 2. UPSERT user ------------------------------------------------
#     extracted_user_id = metadata.get("user_id") or "anonymous"
#     upsert_user(
#         user_id=extracted_user_id,
#         college=metadata.get("college"),
#         role=metadata.get("role"),
#     )

#     # --- 3. INSERT ticket -----------------------------------------------
#     ticket = insert_ticket(
#         user_id=extracted_user_id,
#         ticket_class=metadata.get("ticket_class"),
#         sub_class=metadata.get("sub_class"),
#         severity=metadata.get("severity"),
#         external_ticket_ref=metadata.get("external_ticket_ref"),
#     )

#     # --- 4. INSERT chat_session -----------------------------------------
#     raw_transcript = json.dumps(cleaned, ensure_ascii=False)
#     insert_chat_session(
#         ticket_id=ticket.ticket_id,
#         start_time=start_time,
#         end_time=end_time,
#         ticket_summary=metadata.get("session_summary"),
#         issue_resolved=metadata.get("issue_resolved", False),
#         sentiment=metadata.get("sentiment"),
#         raw_transcript=raw_transcript,
#     )

#     # --- 5. Closure message ---------------------------------------------
#     closure = (
#         "✅ تم إنهاء الجلسة بنجاح وحفظ البيانات.\n\n"
#         f"**رقم التذكرة:** `{ticket.ticket_id}`"
#     )
#     history.append(
#         gr.ChatMessage(
#             role="assistant",
#             content=f'<div dir="auto">\n\n{closure}\n\n</div>',
#         )
#     )

#     return (
#         history,
#         gr.update(interactive=False),   # disable textbox
#         gr.update(interactive=False),   # disable end-session button
#         session_start,
#     )


# def main():
#     def put_message_in_chatbot(message, history):
#         formatted_message = f'<div dir="auto">\n\n{message}\n\n</div>'
#         history.append(gr.ChatMessage(role="user", content=formatted_message))
#         return "", history

#     theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

#     # Check Gradio version behavior dynamically using try-except
#     try:
#         gr.Chatbot(type="messages", show_copy_button=True)
#         is_legacy_gradio = True
#     except TypeError:
#         is_legacy_gradio = False
 
#     blocks_kwargs = {"title": "المساعد التقني لجامعة الملك سعود"}
#     launch_kwargs = {"inbrowser": True}
    
#     if is_legacy_gradio:
#         blocks_kwargs["theme"] = theme
#         blocks_kwargs["css"] = custom_css
#     else:
#         launch_kwargs["theme"] = theme
#         launch_kwargs["css"] = custom_css

#     # تمرير theme و css هنا في Blocks مثل v0.2
#     with gr.Blocks(**blocks_kwargs) as ui:

#         # --- Data Privacy Banner (Section 2) ---
#         gr.Markdown(
#             "> **تنبيه الخصوصية:** يُرجى العلم بأنه يتم تسجيل وتحليل هذه المحادثة "
#             "لأغراض ضمان الجودة وتطوير الخدمات وفقاً لسياسات حماية البيانات."
#         )

#         gr.Markdown("# المساعد التقني لجامعة الملك سعود")

#         # Hidden state to track session start time
#         session_start = gr.State(value=datetime.now(timezone.utc))

#         with gr.Row():
#             with gr.Column(scale=1):
#                 # استخدام show_copy_button بدلاً من buttons
#                 chatbot_kwargs = {
#                     "label": "", 
#                     "height": 600, 
#                     "elem_id": "bidi-chatbot"
#                 }
#                 if is_legacy_gradio:
#                     chatbot_kwargs["show_copy_button"] = True
#                     chatbot_kwargs["type"] = "messages"
                    
#                 chatbot = gr.Chatbot(**chatbot_kwargs)
                
#                 message = gr.Textbox(
#                     label="Your Question",
#                     placeholder="اسألني عن أي مشكلة تقنية وسأحاول مساعدتك",
#                     show_label=False,
#                     elem_id="bidi-input" 
#                 )

#                 # --- End Session button (Section 2) ---
#                 end_session_btn = gr.Button(
#                     "إنهاء الجلسة",
#                     variant="stop",
#                     elem_id="end-session-btn",
#                 )

#         # --- Event wiring ---
#         message.submit(
#             put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
#         ).then(chat, inputs=chatbot, outputs=[chatbot])

#         # End Session pipeline (Section 5)
#         end_session_btn.click(
#             end_session,
#             inputs=[chatbot, session_start],
#             outputs=[chatbot, message, end_session_btn, session_start],
#         )

#     # تشغيل صافي بدون تمرير theme أو css هنا
#     ui.launch(**launch_kwargs)


# if __name__ == "__main__":
#     main()















# import gradio as gr
# from dotenv import load_dotenv
# import re
# import json
# from datetime import datetime, timezone

# from implementation.answer import answer_question
# from implementation.extractor import extract_metadata, build_transcript
# from implementation.database import (
#     init_db,
#     upsert_user,
#     insert_ticket,
#     insert_chat_session,
# )

# load_dotenv(override=True)

# # Initialise the database tables on startup
# init_db()

# custom_css = """
# #bidi-input textarea {
#     unicode-bidi: plaintext;
#     text-align: start;
# }

# #bidi-chatbot .prose p, 
# #bidi-chatbot .prose li, 
# #bidi-chatbot .prose ul, 
# #bidi-chatbot .prose ol {
#     text-align: start !important;
# }

# #chunks-display textarea {
#     font-family: monospace;
#     color: #ffffff !important;
#     background-color: #1e1e1e !important;
# }
# """

# def strip_bidi_tags(text):
#     text = text.replace('<div dir="auto">\n\n', '')
#     text = text.replace('\n\n</div>', '')
#     return text


# def chat(history):
#     # Clean the current message so the AI doesn't see HTML
#     last_msg_content = history[-1]["content"]
#     if isinstance(last_msg_content, list):
#         last_msg_str = "".join([b.get("text", "") for b in last_msg_content if b.get("type") == "text"])
#     else:
#         last_msg_str = str(last_msg_content)
#     last_message = strip_bidi_tags(last_msg_str)
    
#     # Clean the prior history context
#     prior = []
#     for msg in history[:-1]:
#         msg_content = msg["content"]
#         if isinstance(msg_content, list):
#             msg_str = "".join([b.get("text", "") for b in msg_content if b.get("type") == "text"])
#         else:
#             msg_str = str(msg_content)
#         prior.append({"role": msg["role"], "content": strip_bidi_tags(msg_str)})
        
#     answer, context = answer_question(last_message, prior)
    
#     # Automatically add a space after a number and period if it is missing
#     answer = re.sub(r'(?m)^(\s*\d+)\.(?=[^\s])', r'\1. ', answer)
    
#     # Wrap the AI's answer in the BiDi div
#     formatted_answer = f'<div dir="auto">\n\n{answer}\n\n</div>'
#     history.append(gr.ChatMessage(role="assistant", content=formatted_answer))
    
#     # تنسيق الـ Chunks للعرض في النافذة الجانبية
#     chunks_display_text = ""
#     if context:
#         if isinstance(context, list):
#             chunks_display_text = "\n\n========================================\n\n".join(
#                [f"📄 [CHUNK {i+1}]\n{doc.page_content if hasattr(doc, 'page_content') else doc}" for i, doc in enumerate(context)]
#             )
#         else:
#             chunks_display_text = str(context)
#     else:
#         chunks_display_text = "⚠️ لم يتم استرجاع أي سياق (No Chunks Retrieved) لهذا السؤال."

#     # ترجع المحادثة + نص الـ Chunks للنافذة المستقلة
#     return history, chunks_display_text


# def _clean_history_for_extraction(history: list[dict]) -> list[dict]:
#     cleaned = []
#     for msg in history:
#         content = msg["content"]
#         if isinstance(content, list):
#             text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
#         else:
#             text = str(content)
#         cleaned.append({"role": msg["role"], "content": strip_bidi_tags(text)})
#     return cleaned


# def end_session(history, session_start):
#     if not history:
#         history.append(
#             gr.ChatMessage(
#                 role="assistant",
#                 content='<div dir="auto">\n\nلا توجد محادثة لإنهائها.\n\n</div>',
#             )
#         )
#         return (
#             history,
#             gr.update(interactive=False),
#             gr.update(interactive=False),
#             session_start,
#         )

#     end_time = datetime.now(timezone.utc)
#     start_time = session_start or end_time

#     cleaned = _clean_history_for_extraction(history)
#     metadata = extract_metadata(cleaned)

#     extracted_user_id = metadata.get("user_id") or "anonymous"
#     upsert_user(
#         user_id=extracted_user_id,
#         college=metadata.get("college"),
#         role=metadata.get("role"),
#     )

#     ticket = insert_ticket(
#         user_id=extracted_user_id,
#         ticket_class=metadata.get("ticket_class"),
#         sub_class=metadata.get("sub_class"),
#         severity=metadata.get("severity"),
#         external_ticket_ref=metadata.get("external_ticket_ref"),
#     )

#     raw_transcript = json.dumps(cleaned, ensure_ascii=False)
#     insert_chat_session(
#         ticket_id=ticket.ticket_id,
#         start_time=start_time,
#         end_time=end_time,
#         ticket_summary=metadata.get("session_summary"),
#         issue_resolved=metadata.get("issue_resolved", False),
#         sentiment=metadata.get("sentiment"),
#         raw_transcript=raw_transcript,
#     )

#     closure = (
#         "✅ تم إنهاء الجلسة بنجاح وحفظ البيانات.\n\n"
#         f"**رقم التذكرة:** `{ticket.ticket_id}`"
#     )
#     history.append(
#         gr.ChatMessage(
#             role="assistant",
#             content=f'<div dir="auto">\n\n{closure}\n\n</div>',
#         )
#     )

#     return (
#         history,
#         gr.update(interactive=False),
#         gr.update(interactive=False),
#         session_start,
#     )


# def main():
#     def put_message_in_chatbot(message, history):
#         formatted_message = f'<div dir="auto">\n\n{message}\n\n</div>'
#         history.append(gr.ChatMessage(role="user", content=formatted_message))
#         return "", history

#     theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

#     try:
#         gr.Chatbot(type="messages", show_copy_button=True)
#         is_legacy_gradio = True
#     except TypeError:
#         is_legacy_gradio = False

#     blocks_kwargs = {"title": "خبير نظم جامعة الملك سعود"}
#     launch_kwargs = {"inbrowser": True}
    
#     if is_legacy_gradio:
#         blocks_kwargs["theme"] = theme
#         blocks_kwargs["css"] = custom_css
#     else:
#         launch_kwargs["theme"] = theme
#         launch_kwargs["css"] = custom_css

#     with gr.Blocks(**blocks_kwargs) as ui:

#         gr.Markdown(
#             "> **تنبيه الخصوصية:** يُرجى العلم بأنه يتم تسجيل وتحليل هذه المحادثة "
#             "لأغراض ضمان الجودة وتطوير الخدمات وفقاً لسياسات حماية البيانات."
#         )

#         gr.Markdown("# المساعد التقني لجامعة الملك سعود")

#         session_start = gr.State(value=datetime.now(timezone.utc))

#         # تقسيم الشاشة إلى عمودين: المحادثة + نافذة الـ Chunks الجانبية
#         with gr.Row():
#             # العمود الأول: واجهة المحادثة الرئيسية
#             with gr.Column(scale=3):
#                 chatbot_kwargs = {
#                     "label": "", 
#                     "height": 550, 
#                     "elem_id": "bidi-chatbot"
#                 }
#                 if is_legacy_gradio:
#                     chatbot_kwargs["show_copy_button"] = True
#                     chatbot_kwargs["type"] = "messages"
                    
#                 chatbot = gr.Chatbot(**chatbot_kwargs)
                
#                 message = gr.Textbox(
#                     label="Your Question",
#                     placeholder="اسألني عن أي مشكلة تقنية وسأحاول مساعدتك",
#                     show_label=False,
#                     elem_id="bidi-input" 
#                 )

#                 end_session_btn = gr.Button(
#                     "إنهاء الجلسة",
#                     variant="stop",
#                     elem_id="end-session-btn",
#                 )

#             # العمود الثاني: نافذة عرض الـ Chunks المسترجعة من الـ Vector DB
#             with gr.Column(scale=2):
#                 chunks_box = gr.Textbox(
#                     label="النصوص المسترجعة",
#                     placeholder="ستظهر النصوص المسترجعة هنا فور إجابة البوت...",
#                     lines=25,
#                     max_lines=30,
#                     interactive=False,
#                     elem_id="chunks-display"
#                 )

#         # ربط حدث الإرسال بتحديث المحادثة ونافذة الـ Chunks المستقلة
#         message.submit(
#             put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
#         ).then(chat, inputs=chatbot, outputs=[chatbot, chunks_box])

#         end_session_btn.click(
#             end_session,
#             inputs=[chatbot, session_start],
#             outputs=[chatbot, message, end_session_btn, session_start],
#         )

#     ui.launch(**launch_kwargs)


# if __name__ == "__main__":
#     main()






# enhanced version of GUI
import os
import re
import json
from datetime import datetime, timezone
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv

from implementation.answer import answer_question
from implementation.extractor import extract_metadata, build_transcript
from implementation.database import (
    init_db,
    upsert_user,
    insert_ticket,
    insert_chat_session,
)

load_dotenv(override=True)
init_db()

custom_css = """
.gradio-container {
    direction: rtl !important;
    text-align: right !important;
}

#bidi-input textarea {
    direction: rtl !important;
    text-align: right !important;
}

#solution-display {
    border: 1px solid #3b82f6 !important;
    border-radius: 8px !important;
    padding: 16px !important;
    background-color: #1e293b !important;
    color: #ffffff !important;
    margin-top: 12px !important;
    direction: rtl !important;
    text-align: right !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
}

#solution-display h3 {
    color: #60a5fa !important;
    margin-bottom: 8px !important;
}

#solution-display p, #solution-display li {
    color: #f1f5f9 !important;
}

#chunks-display textarea {
    font-family: monospace !important;
    direction: ltr !important;
    text-align: left !important;
}
"""

I18N = {
    "ar": {
        "title": "المساعد التقني لجامعة الملك سعود",
        "privacy": "> **تنبيه الخصوصية:** يُرجى العلم بأنه يتم تسجيل وتحليل هذه المحادثة لأغراض ضمان الجودة وتطوير الخدمات.",
        "catalog_title": "📂 دليل الخدمات (الحلول السريعة)",
        "chat_label": "المحادثة التقنية",
        "chunks_label": "النصوص المسترجعة من قاعدة البيانات (Retrieved Context)",
        "input_placeholder": "اسألني عن أي مشكلة تقنية وسأحاول مساعدتك...",
        "end_session": "إنهاء الجلسة",
        "ask_bot_btn": "💬 لم يحل المشكلة؟ اسأل المساعد",
        "initial_solution": "*اختر إحدى الخدمات من القائمة أعلاه لعرض خطوات استكشاف الأخطاء وإصلاحها السريعة.*",
        "chunks_placeholder": "ستظهر النصوص المسترجعة هنا بعرض كامل فور طرح السؤال في الشات...",
        "btn_acc": "👤 الحسابات والصلاحيات",
        "btn_eml": "📧 البريد الإلكتروني",
        "btn_net": "📶 الشبكات والإنترنت",
        "btn_lms": "📚 نظام إدارة التعلم (Blackboard)",
        "btn_tel": "📞 الخدمات الهاتفية",
        "btn_pcs": "💻 أجهزة الحاسب والبرامج",
    },
    "en": {
        "title": "KSU IT Technical Assistant",
        "privacy": "> **Privacy Notice:** Conversations are logged and analyzed for quality assurance and service improvement.",
        "catalog_title": "📂 Incident Catalog (Quick Solutions)",
        "chat_label": "IT Support Chat",
        "chunks_label": "Retrieved Context Chunks (Full Width)",
        "input_placeholder": "Ask any technical support question...",
        "end_session": "End Session",
        "ask_bot_btn": "💬 Didn't solve it? Ask Assistant",
        "initial_solution": "*Select a service category above to view quick troubleshooting steps.*",
        "chunks_placeholder": "Retrieved vector context will appear here in full width upon asking...",
        "btn_acc": "👤 Accounts & Permissions",
        "btn_eml": "📧 KSU Email",
        "btn_net": "📶 Networks & Internet",
        "btn_lms": "📚 Learning Management (LMS)",
        "btn_tel": "📞 Telephony Services",
        "btn_pcs": "💻 PCs & Software",
    }
}

def strip_bidi_tags(text):
    text = text.replace('<div dir="auto">\n\n', '')
    text = text.replace('\n\n</div>', '')
    return text

def chat(history):
    last_msg_content = history[-1]["content"]
    if isinstance(last_msg_content, list):
        last_msg_str = "".join([b.get("text", "") for b in last_msg_content if b.get("type") == "text"])
    else:
        last_msg_str = str(last_msg_content)
    last_message = strip_bidi_tags(last_msg_str)
    
    prior = []
    for msg in history[:-1]:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            msg_str = "".join([b.get("text", "") for b in msg_content if b.get("type") == "text"])
        else:
            msg_str = str(msg_content)
        prior.append({"role": msg["role"], "content": strip_bidi_tags(msg_str)})
        
    answer, context = answer_question(last_message, prior)
    answer = re.sub(r'(?m)^(\s*\d+)\.(?=[^\s])', r'\1. ', answer)
    
    formatted_answer = f'<div dir="auto">\n\n{answer}\n\n</div>'
    history.append(gr.ChatMessage(role="assistant", content=formatted_answer))
    
    chunks_display_text = ""
    if context:
        if isinstance(context, list):
            formatted_chunks = []
            for i, doc in enumerate(context):
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                source_path = doc.metadata.get('source', '') if hasattr(doc, 'metadata') else ''
                source_name = Path(source_path).name if source_path else 'N/A'
                
                header = f"📄 [CHUNK {i+1}] | Source: {source_name}"
                divider = "-" * 80
                formatted_chunks.append(f"{header}\n{divider}\n{content}")
            
            chunks_display_text = "\n\n================================================================================\n\n".join(formatted_chunks)
        else:
            chunks_display_text = str(context)
    else:
        chunks_display_text = "⚠️ لم يتم استرجاع أي سياق لهذا السؤال."

    return history, chunks_display_text

def end_session(history, session_start):
    if not history:
        history.append(gr.ChatMessage(role="assistant", content='<div dir="auto">\n\nلا توجد محادثة لإنهائها.\n\n</div>'))
        return history, gr.update(interactive=False), gr.update(interactive=False), session_start

    end_time = datetime.now(timezone.utc)
    start_time = session_start or end_time

    cleaned = []
    for msg in history:
        content = msg["content"]
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text") if isinstance(content, list) else str(content)
        cleaned.append({"role": msg["role"], "content": strip_bidi_tags(text)})

    metadata = extract_metadata(cleaned)
    extracted_user_id = metadata.get("user_id") or "anonymous"
    upsert_user(user_id=extracted_user_id, college=metadata.get("college"), role=metadata.get("role"))

    ticket = insert_ticket(
        user_id=extracted_user_id,
        ticket_class=metadata.get("ticket_class"),
        sub_class=metadata.get("sub_class"),
        severity=metadata.get("severity"),
        external_ticket_ref=metadata.get("external_ticket_ref"),
    )

    insert_chat_session(
        ticket_id=ticket.ticket_id,
        start_time=start_time,
        end_time=end_time,
        ticket_summary=metadata.get("session_summary"),
        issue_resolved=metadata.get("issue_resolved", False),
        sentiment=metadata.get("sentiment"),
        raw_transcript=json.dumps(cleaned, ensure_ascii=False),
    )

    closure = f"✅ تم إنهاء الجلسة بنجاح وحفظ البيانات.\n\n**رقم التذكرة:** `{ticket.ticket_id}`"
    history.append(gr.ChatMessage(role="assistant", content=f'<div dir="auto">\n\n{closure}\n\n</div>'))

    return history, gr.update(interactive=False), gr.update(interactive=False), session_start

def main():
    def put_message_in_chatbot(message, history):
        formatted_message = f'<div dir="auto">\n\n{message}\n\n</div>'
        history.append(gr.ChatMessage(role="user", content=formatted_message))
        return "", history

    def show_quick_solution(category_key, lang):
        solutions_ar = {
            "accounts": "### 🔑 الحسابات والصلاحيات\n\n**خطوات استكشاف الأخطاء وإصلاحها:**\n- التأكد من إدخال اسم المستخدم وكلمة المرور بشكل صحيح.\n- استخدام الخدمة الذاتية لإعادة تعيين كلمة المرور عبر البوابة.\n- التحقق من تفعيل تطبيق التحقق الثنائي (MFA).",
            "email": "### 📧 البريد الإلكتروني\n\n**خطوات استكشاف الأخطاء وإصلاحها:**\n- التأكد من الاتصال بالإنترنت قبل فتح Outlook.\n- التحقق من المساحة المتاحة في صندوق البريد.\n- إعادة إدخال بيانات الاعتماد عند تغيير كلمة المرور.",
            "networks": "### 📶 الشبكات والإنترنت\n\n**خطوات استكشاف الأخطاء وإصلاحها:**\n- التأكد من التواجد في نطاق تغطية شبكة الجامعة اللاسلكية.\n- تسجيل الدخول باستخدام الحساب الجامعي.\n- إلغاء حفظ الشبكة (Forget Network) ثم إعادة الاتصال.",
            "lms": "### 📚 نظام إدارة التعلم (Blackboard)\n\n**خطوات استكشاف الأخطاء وإصلاحها:**\n- استخدام متصفح حديث ومسح الـ Cache.\n- تسجيل الدخول عبر نظام الدخول الموحد (SSO).\n- التأكد من تفعيل المقررات في الجدول.",
            "telephony": "### 📞 الخدمات الهاتفية\n\n**خطوات استكشاف الأخطاء وإصلاحها:**\n- التأكد من توصيل كابل الهاتف الشبكي.\n- التحقق من توفر الخدمة والتحويلات الداخلية.",
            "pcs": "### 💻 أجهزة الحاسب والبرامج\n\n**خطوات استكشاف الأخطاء وإصلاحها:**\n- إعادة تشغيل الجهاز قبل رفع البلاغ.\n- التحقق من التحديثات وتوصيلات الطاقة والشبكة."
        }
        
        solutions_en = {
            "accounts": "### 🔑 Accounts & Permissions\n\n**Troubleshooting Steps:**\n- Verify username and password credentials.\n- Use Self-Service Password Reset portal if needed.\n- Ensure Multi-Factor Authentication (MFA) app is verified.",
            "email": "### 📧 KSU Email\n\n**Troubleshooting Steps:**\n- Check internet connection before launching Outlook.\n- Verify mailbox storage capacity.\n- Re-enter credentials if password changed recently.",
            "networks": "### 📶 Networks & Internet\n\n**Troubleshooting Steps:**\n- Ensure you are within KSU wireless coverage.\n- Log in using your valid university credentials.\n- Forget network and reconnect if authentication fails.",
            "lms": "### 📚 Learning Management System (LMS)\n\n**Troubleshooting Steps:**\n- Use Chrome/Edge and clear browser cache.\n- Authenticate through Single Sign-On (SSO).\n- Confirm course enrollment in academic schedule.",
            "telephony": "### 📞 Telephony Services\n\n**Troubleshooting Steps:**\n- Check IP Phone network cable connections.\n- Verify line availability and internal extension setup.",
            "pcs": "### 💻 PCs & Software\n\n**Troubleshooting Steps:**\n- Restart computer before submitting support requests.\n- Ensure system updates and approved software are installed."
        }

        dict_ref = solutions_ar if lang == "ar" else solutions_en
        return dict_ref.get(category_key, ""), gr.update(visible=True)

    def ask_chatbot_from_solution(solution_text, history):
        if not solution_text:
            return history
        first_line = solution_text.split("\n")[0].replace("#", "").strip()
        query = f"أحتاج مساعدة إضافية في: {first_line}"
        formatted_message = f'<div dir="auto">\n\n{query}\n\n</div>'
        history.append(gr.ChatMessage(role="user", content=formatted_message))
        return history

    def change_language(lang):
        texts = I18N[lang]
        return (
            texts["title"],
            texts["privacy"],
            texts["catalog_title"],
            texts["btn_acc"],
            texts["btn_eml"],
            texts["btn_net"],
            texts["btn_lms"],
            texts["btn_tel"],
            texts["btn_pcs"],
            texts["initial_solution"],
            texts["ask_bot_btn"],
            gr.update(label=texts["chat_label"]),
            gr.update(placeholder=texts["input_placeholder"]),
            texts["end_session"],
            gr.update(label=texts["chunks_label"], placeholder=texts["chunks_placeholder"])
        )

    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=theme, css=custom_css, title="المساعد التقني - جامعة الملك سعود") as ui:

        current_lang = gr.State(value="ar")
        session_start = gr.State(value=datetime.now(timezone.utc))

        with gr.Row():
            title_md = gr.Markdown("# المساعد التقني لجامعة الملك سعود")
            lang_dropdown = gr.Dropdown(
                choices=[("العربية", "ar"), ("English", "en")],
                value="ar",
                label="اللغة / Language",
                interactive=True,
                scale=0
            )

        privacy_md = gr.Markdown(I18N["ar"]["privacy"])

        with gr.Row():
            # ⬅️ العمود الأيسر: المحادثة التقنية (الشات بوت)
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label=I18N["ar"]["chat_label"], height=450, elem_id="bidi-chatbot")
                message = gr.Textbox(placeholder=I18N["ar"]["input_placeholder"], show_label=False, elem_id="bidi-input")
                end_session_btn = gr.Button(I18N["ar"]["end_session"], variant="stop")

            # ➡️ العمود الأيمن: دليل الخدمات والحلول السريعة (الخدمات على اليسار بصرية)
            with gr.Column(scale=2):
                catalog_title = gr.Markdown("### 📂 دليل الخدمات (الحلول السريعة)")
                
                btn_acc = gr.Button(I18N["ar"]["btn_acc"])
                btn_eml = gr.Button(I18N["ar"]["btn_eml"])
                btn_net = gr.Button(I18N["ar"]["btn_net"])
                btn_lms = gr.Button(I18N["ar"]["btn_lms"])
                btn_tel = gr.Button(I18N["ar"]["btn_tel"])
                btn_pcs = gr.Button(I18N["ar"]["btn_pcs"])

                solution_box = gr.Markdown(value=I18N["ar"]["initial_solution"], elem_id="solution-display")

                with gr.Row(visible=False) as action_row:
                    btn_ask_bot = gr.Button(I18N["ar"]["ask_bot_btn"], variant="primary")

        # الجزء السفلي: نافذة الاسترجاع الممتدة بعرض القاع كاملاً
        with gr.Row():
            chunks_box = gr.Textbox(
                label=I18N["ar"]["chunks_label"],
                placeholder=I18N["ar"]["chunks_placeholder"],
                lines=10,
                max_lines=15,
                interactive=False,
                elem_id="chunks-display"
            )

        # الأحداث والربط
        lang_dropdown.change(
            fn=lambda l: (l, *change_language(l)),
            inputs=[lang_dropdown],
            outputs=[
                current_lang, title_md, privacy_md, catalog_title,
                btn_acc, btn_eml, btn_net, btn_lms, btn_tel, btn_pcs,
                solution_box, btn_ask_bot, chatbot, message, end_session_btn, chunks_box
            ]
        )

        btn_acc.click(fn=lambda l: show_quick_solution("accounts", l), inputs=[current_lang], outputs=[solution_box, action_row])
        btn_eml.click(fn=lambda l: show_quick_solution("email", l), inputs=[current_lang], outputs=[solution_box, action_row])
        btn_net.click(fn=lambda l: show_quick_solution("networks", l), inputs=[current_lang], outputs=[solution_box, action_row])
        btn_lms.click(fn=lambda l: show_quick_solution("lms", l), inputs=[current_lang], outputs=[solution_box, action_row])
        btn_tel.click(fn=lambda l: show_quick_solution("telephony", l), inputs=[current_lang], outputs=[solution_box, action_row])
        btn_pcs.click(fn=lambda l: show_quick_solution("pcs", l), inputs=[current_lang], outputs=[solution_box, action_row])

        btn_ask_bot.click(
            ask_chatbot_from_solution, inputs=[solution_box, chatbot], outputs=[chatbot]
        ).then(chat, inputs=[chatbot], outputs=[chatbot, chunks_box])

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=[chatbot], outputs=[chatbot, chunks_box])

        end_session_btn.click(
            end_session, inputs=[chatbot, session_start], outputs=[chatbot, message, end_session_btn, session_start]
        )

    ui.launch(inbrowser=True)

if __name__ == "__main__":
    main()