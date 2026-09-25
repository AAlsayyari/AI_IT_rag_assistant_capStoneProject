# Ticketing System Analytics & Backend Architecture (v0.3)

## 1. Development & Version Control
To ensure the stability of the existing RAG integration, all new analytics and database features are isolated in a new build environment (`v0.3`). This acts as a safeguard, leaving the working `v0.2` application untouched while the database ingestion pipeline and UI state changes are tested.

## 2. User Interface Modifications (Gradio)
The frontend requires two primary updates to handle compliance and session lifecycle management:

* **Data Privacy Banner:** A static Markdown banner is injected at the top of the UI (`> **تنبيه الخصوصية:** يُرجى العلم بأنه يتم تسجيل وتحليل هذه المحادثة لأغراض ضمان الجودة وتطوير الخدمات وفقاً لسياسات حماية البيانات.`). This ensures transparency and complies with standard data privacy practices before any chat data is permanently logged.
* **Session Lifecycle Controls:** An "End Session" (`إنهاء الجلسة`) button triggers the backend pipeline. Upon click, the text input box is immediately disabled. This state freeze prevents the user from sending new messages while the backend is processing the summarization and database injection, ensuring the analyzed data perfectly matches the final state of the conversation.

## 3. Data Extraction Engine
Rather than storing raw chat transcripts as flat text, the system utilizes a Large Language Model (LLM) configured for Structured Output (JSON) to parse the transcript and extract operational metadata upon session closure.

The extraction pipeline captures:
* **Ticket Classification:** The primary category (e.g., Hardware, LMS, Network) and specific sub-class (e.g., Password Reset, Wi-Fi Certificate) for routing.
* **Severity & Sentiment:** An analysis of the issue's urgency and the user's frustration level, enabling future dashboarding on user satisfaction and critical system failures.
* **Resolution Status:** A boolean flag determining if the AI successfully deflected the issue or if human intervention is required. 
* **Session Summary:** A concise, 2-3 sentence recap of the problem and the AI's troubleshooting steps, optimizing the workflow for human agents handling escalations.

## 4. Relational Database Architecture
To support future Business Intelligence (BI) tools and dashboarding, the data is stored in a normalized relational database (SQLite/PostgreSQL) split into three linked tables:

### `users` Table
Stores organizational context to prevent duplicating attributes across multiple tickets.
* `user_id` (Primary Key): Unique identifier.
* `college`: Academic college or department.
* `role`: User classification (e.g., student, faculty).

### `tickets` Table
Captures the overarching issue context.
* `ticket_id` (Primary Key): UUID.
* `user_id` (Foreign Key): Links to the `users` table.
* `external_ticket_ref`: Any existing ticket number provided by the user.
* `ticket_class` & `sub_class`: Categorization fields.
* `severity`: Extracted urgency level.
* `created_at`: Initialization timestamp.

### `chat_sessions` Table
Records the actual interaction metrics and links to the parent ticket.
* `session_id` (Primary Key): UUID.
* `ticket_id` (Foreign Key): Links to the `tickets` table.
* `start_time` & `end_time`: Timestamps for session duration metrics.
* `ticket_summary`: The LLM-generated recap.
* `issue_resolved`: Boolean deflection metric.
* `sentiment`: Extracted user mood.
* `raw_transcript`: The full JSON/text dump for auditing and future RAG retraining.

## 5. Event Loop Integration
The backend pipeline connects directly to the Gradio event loop. When the "End Session" trigger fires, the pipeline extracts the metadata, executes the necessary `INSERT/UPSERT` operations across the three tables using a database ORM (like SQLAlchemy), and returns a clean closure message to the UI without interrupting the core application state.