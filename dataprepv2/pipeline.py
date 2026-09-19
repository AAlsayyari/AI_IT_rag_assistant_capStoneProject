import os
from pathlib import Path
from openai import OpenAI
import customtkinter as ctk
import threading
import re

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat

# ==============================================================================
# Configuration & Setup
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_MD_DIR = BASE_DIR / "raw_md"
CLEAN_MD_DIR = BASE_DIR / "clean_markdown"

for directory in [DATASET_DIR, RAW_MD_DIR, CLEAN_MD_DIR]:
    os.makedirs(directory, exist_ok=True)

import dotenv

env_path = BASE_DIR.parent / ".env"
if env_path.exists():
    dotenv.load_dotenv(dotenv_path=env_path, override=True)

try:
    openai_client = OpenAI()
except Exception as e:
    openai_client = None

# ==============================================================================
# Prompts
# ==============================================================================

OPENAI_SYSTEM_PROMPT = """You are an elite Enterprise Data Architect and Strict Technical Extractor building a pristine Arabic RAG database. Your ONLY job is to convert messy OCR text into a rigid, highly structured Markdown format without losing a single character of business logic.

You must process the text applying the following ABSOLUTE rules:

### 1. STRICT TEMPLATE ENFORCEMENT (THE CURE FOR SUMMARIZATION)
You are strictly forbidden from writing free-form summaries. You MUST format EVERY system screen, process, or major topic using this EXACT structural template:

## [Name of the Screen or Topic]
### الوصف
[Extract the precise description of what this screen/feature does. Do NOT summarize.]
### الخطوات
1. [Step 1]
2. [Step 2]
...

*CRITICAL RULE:* If the original OCR text contains a numbered list, bullet points, or sequential instructions, you MUST populate the `### الخطوات` section. If and ONLY if there are genuinely zero steps in the source text for that section, write "لا يوجد خطوات".

### 2. THE ZERO-DROP MANDATE (NO DATA LOSS)
- Do NOT condense, paraphrase, or summarize. 
- You must actively hunt for fragmented lists in the OCR text and reconstruct them into the template. 
- You MUST retain every single business rule, dropdown option, condition, warning, system parameter, character limit, and exact URL verbatim. (e.g., If the text explains 3 privacy levels, you must extract the explanation for all 3 levels).

### 3. UI RECONSTRUCTION (THE "HANGING VERB" FIX)
- The OCR engine frequently drops inline UI icons, leaving hanging verbs (e.g., "اضغط على للدخول", "اختر لتعديل").
- You MUST logically infer the missing button or icon name from context and insert it in brackets (e.g., "اضغط على [دخول] للدخول", "اختر [تعديل]"). NEVER output a verb without its target object.

### 4. DESTROY NON-INFORMATIONAL CONTENT & VISUALS
- Completely REMOVE the Table of Contents (الفهرس, المحتويات), page numbers, headers/footers, and copyright notices.
- REMOVE all visual, spatial, and image-based references (e.g., "كما في الشاشة أعلاه", "المنطقة 1", "انظر الشكل"). Convert visual descriptions into direct instructional statements.

OUTPUT FORMAT: Output ONLY the finalized Markdown text. Do not include any conversational filler. Start immediately with the first `##` header."""
# ==============================================================================
# Utility Functions
# ==============================================================================

def extract_number(path):
    """Extracts the first sequence of digits from a filename for natural sorting."""
    match = re.search(r'\d+', path.name)
    return int(match.group()) if match else float('inf')

# ==============================================================================
# Modern UI Application
# ==============================================================================

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class PipelineApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("High-Fidelity RAG Data Prep Pipeline")
        self.geometry("900x650")
        
        self.stop_event = threading.Event()
        self.current_thread = None
        
        # Grid Layout
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Top Frame for Controls
        self.control_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.control_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        self.control_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Phase 1 Button
        self.btn_phase1 = ctk.CTkButton(
            self.control_frame, 
            text="1. Extract PDFs (Docling)", 
            command=self.start_phase1,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.btn_phase1.grid(row=0, column=0, padx=10, sticky="ew")
        
        # Stop Button
        self.btn_stop = ctk.CTkButton(
            self.control_frame, 
            text="■ STOP", 
            command=self.stop_process,
            fg_color="#D32F2F", 
            hover_color="#B71C1C",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled"
        )
        self.btn_stop.grid(row=0, column=1, padx=10, sticky="ew")
        
        # Phase 2 Button
        self.btn_phase2 = ctk.CTkButton(
            self.control_frame, 
            text="2. Format to Markdown", 
            command=self.start_phase2,
            fg_color="#2E7D32", 
            hover_color="#1B5E20",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.btn_phase2.grid(row=0, column=2, padx=10, sticky="ew")
        
        # Log Text Area
        self.log_area = ctk.CTkTextbox(self, font=ctk.CTkFont(family="Consolas", size=13), corner_radius=10)
        self.log_area.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        
        self.log_message("🚀 System initialized. Ready to process.")
        self.log_message("ℹ️ Note: Stop command will interrupt before the next page/file is sent to the API.")

    def log_message(self, message):
        # Safely interact with Tkinter from another thread
        self.after(0, self._append_log, message)
        
    def _append_log(self, message):
        self.log_area.insert("end", message + "\n")
        self.log_area.see("end")

    def set_ui_state(self, running):
        if running:
            self.btn_phase1.configure(state="disabled")
            self.btn_phase2.configure(state="disabled")
            self.btn_stop.configure(state="normal")
        else:
            self.btn_phase1.configure(state="normal")
            self.btn_phase2.configure(state="normal")
            self.btn_stop.configure(state="disabled")

    def stop_process(self):
        self.log_message("\n[System] 🛑 STOPPING... The pipeline will halt after the current operation finishes.")
        self.stop_event.set()

    def start_phase1(self):
        self.stop_event.clear()
        self.set_ui_state(running=True)
        self.current_thread = threading.Thread(target=self.run_phase1, daemon=True)
        self.current_thread.start()

    def start_phase2(self):
        self.stop_event.clear()
        self.set_ui_state(running=True)
        self.current_thread = threading.Thread(target=self.run_phase2, daemon=True)
        self.current_thread.start()

    def run_phase1(self):
        try:
            self.log_message("\n" + "━"*60)
            self.log_message("=== Phase 1: Native PDF Parsing & OCR with Docling ===")
            self.log_message("━"*60)
            
            # Fetch and sort files by the numbers in their names
            pdf_files = sorted(list(Path(DATASET_DIR).glob("*.pdf")), key=extract_number)
            
            if not pdf_files:
                self.log_message(f"[!] No PDFs found in '{DATASET_DIR}'. Please add PDFs and try again.")
                return

            self.log_message(f"📂 Found {len(pdf_files)} PDFs. Processing in numerical order...")

            # Initialize Docling Converter
            self.log_message("-> Initializing Docling OCR Engine (EasyOCR for Arabic/English)...")
            pipeline_options = PdfPipelineOptions(do_ocr=True)
            pipeline_options.ocr_options = EasyOcrOptions(lang=["ar", "en"])
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            for pdf_path in pdf_files:
                if self.stop_event.is_set():
                    self.log_message("[System] Phase 1 stopped by user.")
                    return

                raw_md_path = Path(RAW_MD_DIR) / f"{pdf_path.stem}.md"
                if raw_md_path.exists():
                    self.log_message(f"[-] Skipping '{pdf_path.name}', already extracted.")
                    continue

                self.log_message(f"\n📄 [Phase 1] Processing PDF: {pdf_path.name}")
                try:
                    self.log_message("  -> Running Docling extraction...")
                    result = converter.convert(str(pdf_path))
                    
                    if self.stop_event.is_set():
                        self.log_message("[System] Phase 1 stopped by user during conversion.")
                        return

                    self.log_message("  -> Exporting to Markdown...")
                    md_text = result.document.export_to_markdown()
                    
                    with open(raw_md_path, "w", encoding="utf-8") as f:
                        f.write(md_text)
                    self.log_message(f"  [+] Saved raw markdown to {raw_md_path.name}")

                except Exception as e:
                    self.log_message(f"  [!] Error processing PDF {pdf_path.name}: {e}")
                    continue
            
            self.log_message("\n" + "━"*60)
            self.log_message("✅ Phase 1 Complete")
            self.log_message("All PDFs have been processed and saved to the 'raw_md' folder.")
            self.log_message("👉 INSTRUCTION: Please press '2. Format to Markdown' to start Phase 2.")
            self.log_message("━"*60 + "\n")
            
        except Exception as e:
            self.log_message(f"[!] Critical Error in Phase 1: {e}")
        finally:
            self.after(0, self.set_ui_state, False)

    def run_phase2(self):
        try:
            if not openai_client:
                self.log_message("[!] OpenAI client not initialized. Ensure OPENAI_API_KEY is set in environment.")
                return

            self.log_message("\n" + "━"*60)
            self.log_message("=== Phase 2: Formatting with OpenAI ===")
            self.log_message("━"*60)
            
            raw_files = sorted(list(Path(RAW_MD_DIR).glob("*.md")), key=extract_number)
            
            if not raw_files:
                self.log_message(f"[!] No raw markdown files found in '{RAW_MD_DIR}'. Please run Phase 1 first.")
                return

            self.log_message(f"📂 Found {len(raw_files)} Markdown files. Processing in numerical order...")

            for raw_path in raw_files:
                if self.stop_event.is_set():
                    self.log_message("[System] Phase 2 stopped by user.")
                    return

                clean_md_path = Path(CLEAN_MD_DIR) / f"{raw_path.stem}.md"
                if clean_md_path.exists():
                    self.log_message(f"[-] Skipping '{raw_path.name}', already formatted.")
                    continue

                self.log_message(f"\n📝 [Phase 2] Formatting: {raw_path.name}")
                with open(raw_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()

                if not raw_text.strip():
                    self.log_message("  [-] Raw text is empty. Skipping.")
                    continue

                self.log_message("  -> Sending text to GPT for strict formatting...")
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                            {"role": "user", "content": raw_text}
                        ],
                        temperature=0.1
                    )
                    clean_md = response.choices[0].message.content.strip()
                    
                    with open(clean_md_path, "w", encoding="utf-8") as f:
                        f.write(clean_md)
                    self.log_message(f"  [+] Saved clean markdown to {clean_md_path.name}")
                except Exception as e:
                    self.log_message(f"  [!] Error formatting text with OpenAI: {e}")

            self.log_message("\n" + "━"*60)
            self.log_message("✅ Phase 2 Complete")
            self.log_message("All Markdown files are ready and saved in 'clean_markdown' folder.")
            self.log_message("━"*60 + "\n")
        
        except Exception as e:
            self.log_message(f"[!] Critical Error in Phase 2: {e}")
        finally:
            self.after(0, self.set_ui_state, False)

if __name__ == "__main__":
    app = PipelineApp()
    app.mainloop()
