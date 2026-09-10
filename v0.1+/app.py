import gradio as gr
from dotenv import load_dotenv

from implementation.answer import answer_question

load_dotenv(override=True)


# def format_context(context):
#     result = "<h2 style='color: #ff7800;'>السياق ذات الصلة</h2>\n\n"
#     for doc in context:
#         result += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
#         result += doc.page_content + "\n\n"
#     return result


def extract_text(content):
    if isinstance(content, list):
        return " ".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
    return content


def chat(history):
    last_message = extract_text(history[-1]["content"])
    prior = [{"role": m["role"], "content": extract_text(m["content"])} for m in history[:-1]]
    answer, context = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    # return history, format_context(context)
    return history


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="خبير نظم جامعة الملك سعود") as ui:
        gr.Markdown("# خبير نظم جامعة الملك سعود")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation", height=600, buttons=["copy"], rtl=True
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="اسألني عن أي مشكلة تقنية في نظم جامعة الملك سعود وسأحاول مساعدتك",
                    show_label=False,
                )

           

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot])

    ui.launch(inbrowser=True, auth=("KSU","RAG"), share=True, theme=theme)


if __name__ == "__main__":
    main()
