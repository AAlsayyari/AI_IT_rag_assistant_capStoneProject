import gradio as gr
from dotenv import load_dotenv

from implementation.answer import answer_question

load_dotenv(override=True)

# 1. تعريف الـ CSS المخصص لضبط الاتجاه RTL
custom_css = """
.message, .bot-message, .user-message, .markdown-text {
    direction: rtl !important;
    text-align: right !important;
}
.message p, .bot-message p, .user-message p {
    direction: rtl !important;
    text-align: right !important;
}
/* إصلاح محاذاة النقاط والقوائم */
ul, ol {
    padding-right: 20px !important;
    padding-left: 0px !important;
    direction: rtl !important;
}
"""



# CSS ذكي يكتشف لغة النص تلقائياً (RTL للعربي و LTR للإنجليزي)
# custom_css = """
# .message, .bot-message, .user-message, .markdown-text, .message p {
#     unicode-bidi: plaintext !important;
#     text-align: start !important;
# }

# /* ضبط محاذاة القوائم لتتكيف مع الاتجاه */
# ul, ol {
#     unicode-bidi: plaintext !important;
#     padding-inline-start: 25px !important;
# }
# """



# CSS هجين يضمن عرض العربي من اليمين والإنجليزي من اليسار بدقة
# custom_css = """
# /* تحديد الاتجاه التلقائي لجميع نصوص الرسائل */
# .message, .bot-message, .user-message, .markdown-text, .message p, .message div {
#     dir: auto !important;
#     direction: auto !important;
#     unicode-bidi: plaintext !important;
#     text-align: initial !important;
# }

# /* ضبط القوائم النقطية والرقمية لتتحاذى تلقائياً مع لغة الفقرة */
# .markdown-text ul, .markdown-text ol {
#     unicode-bidi: plaintext !important;
#     padding-inline-start: 2rem !important;
#     margin-inline-start: 0px !important;
# }

# /* محاذاة عناصر القائمة لليمين في حالة النص العربي */
# .markdown-text li {
#     unicode-bidi: plaintext !important;
#     text-align: start !important;
# }
# """




def chat(history):
    last_message = history[-1]["content"]
    prior = history[:-1]
    answer, context = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    # 2. تمرير custom_css داخل gr.Blocks هنا
    with gr.Blocks(title="خبير نظم جامعة الملك سعود", theme=theme, css=custom_css) as ui:
        gr.Markdown("# خبير نظم جامعة الملك سعود")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="", height=600, type="messages", show_copy_button=True
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="اسألني عن أي مشكلة تقنية في نظم جامعة الملك سعود وسأحاول مساعدتك",
                    show_label=False,
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot])

    ui.launch(inbrowser=True, auth=("KSU", "RAG"), share=True)


if __name__ == "__main__":
    main()










