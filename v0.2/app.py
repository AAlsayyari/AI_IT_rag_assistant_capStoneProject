import gradio as gr
from dotenv import load_dotenv
import re

from implementation.answer import answer_question

load_dotenv(override=True)

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
    last_message = strip_bidi_tags(history[-1]["content"])
    
    # Clean the prior history context
    prior = []
    for msg in history[:-1]:
        prior.append({"role": msg["role"], "content": strip_bidi_tags(msg["content"])})
        
    answer, context = answer_question(last_message, prior)
    
    # Automatically add a space after a number and period if it is missing (1.افتح -> 1. افتح)
    answer = re.sub(r'(?m)^(\s*\d+)\.(?=[^\s])', r'\1. ', answer)
    
    # Wrap the AI's answer in the BiDi div to natively fix punctuation and list markers
    formatted_answer = f'<div dir="auto">\n\n{answer}\n\n</div>'
    
    history.append({"role": "assistant", "content": formatted_answer})
    return history


def main():
    def put_message_in_chatbot(message, history):
        # Wrap the user's message in the BiDi div as well
        formatted_message = f'<div dir="auto">\n\n{message}\n\n</div>'
        return "", history + [{"role": "user", "content": formatted_message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="خبير نظم جامعة الملك سعود", theme=theme, css=custom_css) as ui:
        gr.Markdown("# خبير نظم جامعة الملك سعود")

        with gr.Row():
            with gr.Column(scale=1):
                # 2. Add the elem_id back so the CSS can target it
                chatbot = gr.Chatbot(
                    label="💬 Conversation", 
                    height=600, 
                    type="messages", 
                    show_copy_button=True,
                    elem_id="bidi-chatbot"
                )
                
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="اسألني عن أي مشكلة تقنية في نظم جامعة الملك سعود وسأحاول مساعدتك",
                    show_label=False,
                    elem_id="bidi-input" 
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot])

    ui.launch(inbrowser=True, auth=("KSU","RAG"), share=True)


if __name__ == "__main__":
    main()