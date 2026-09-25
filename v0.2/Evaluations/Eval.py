import os
import sys
import math
import json
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from litellm import acompletion, completion
from dotenv import load_dotenv

# إضافة المجلد الرئيسي والمجلد الحالي لمسارات بايثون
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(__file__))

# استيراد نموذج البيانات ودالة التحميل من Test.py
from Test import TestQuestion, load_tests
# استيراد دوال المشروع الرئيسي
from implementation.answer import answer_question, fetch_context

load_dotenv(override=True)
# qwen2.5:14b
# gemma2:9b
JUDGE_MODEL = "gemma2:9b"  # نموذج LLM لتقييم الإجابات
EVAL_DIR = Path(__file__).parent
DATASET_FILE = str(EVAL_DIR / "tests.jsonl")
MAX_CONCURRENT_TASKS = 2


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""
    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""
    feedback: str = Field(
        description="ملاحظات دقيقة وموجزة حول جودة الإجابة مقارنة بالإجابة النموذجية."
    )
    accuracy: float = Field(
        description="مدى صحة الإجابة مقارنة بالإجابة النموذجية (من 1 إلى 5). الإجابة الخاطئة تأخذ 1."
    )
    completeness: float = Field(
        description="مدى شمولية الإجابة وتغطيتها لجميع جوانب الإجابة النموذجية (من 1 إلى 5)."
    )
    relevance: float = Field(
        description="مدى ارتباط الإجابة بالسؤال بشكل مباشر دون إضافات غير ضرورية (من 1 إلى 5)."
    )



import re

def clean_text(text: str) -> str:
    """
    تنظيف النص من رموز التنسيق وتوحيد الحروف العربية 
    لضمان مطابقة الكلمات المفتاحية بدقة عالية متجاوزة اختلافات الكتابة.
    """
    if not text:
        return ""
        
    # 1. إزالة رموز Markdown وعلامات الترقيم
    text = re.sub(r'[\*\#\-\_\=\|\[\]\(\)\`\>]', ' ', text)
    text = re.sub(r'[,\.\:\؟\?\,\؛\;\"\'\«\»]', ' ', text)
    
    # 2. إزالة التشكيل (الحركات) والتطويل
    text = re.sub(r'[\u064B-\u065F\u0640]', '', text)
    
    # 3. معالجة الحروف المركبة (Decomposed Unicode) التي تظهر أحياناً في استخراج النصوص
    text = text.replace('\u064A\u0654', 'ئ') # ي + همزة
    text = text.replace('\u0627\u0654', 'أ') # ا + همزة
    text = text.replace('\u0627\u0655', 'إ') # ا + همزة سفلية
    
    # 4. توحيد الحروف العربية (Normalization)
    text = re.sub(r'[إأآا]', 'ا', text)    # توحيد الألف
    text = re.sub(r'ة', 'ه', text)         # توحيد التاء المربوطة والهاء
    text = re.sub(r'[يى]', 'ى', text)      # توحيد الياء والألف المقصورة
    text = re.sub(r'ؤ', 'و', text)         # توحيد الواو المهموزة
    text = re.sub(r'ئ', 'ى', text)         # توحيد النبرة
    
    # 5. توحيد المسافات (بما فيها الأسطر الجديدة) وتحويل الحروف الإنجليزية للصغيرة
    return ' '.join(text.split()).lower()


# def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
#     keyword_lower = keyword.lower()
#     for rank, doc in enumerate(retrieved_docs, start=1):
#         if keyword_lower in doc.page_content.lower():
#             return 1.0 / rank
#     return 0.0


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """حساب MRR بعد تنظيف الكلمات المفتاحية ونصوص المستندات من Markdown"""
    clean_keyword = clean_text(keyword)
    if not clean_keyword:
        return 0.0

    for rank, doc in enumerate(retrieved_docs, start=1):
        clean_doc = clean_text(doc.page_content)
        if clean_keyword in clean_doc:
            return 1.0 / rank
    return 0.0

def calculate_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg



# def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
#     keyword_lower = keyword.lower()
#     relevances = [1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]]
#     dcg = calculate_dcg(relevances, k)
#     ideal_relevances = sorted(relevances, reverse=True)
#     idcg = calculate_dcg(ideal_relevances, k)
#     return dcg / idcg if idcg > 0 else 0.0


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    """حساب nDCG بعد تنظيف النصوص لتفادي الرموز الخاصة"""
    clean_keyword = clean_text(keyword)
    if not clean_keyword:
        return 0.0

    relevances = []
    for doc in retrieved_docs[:k]:
        clean_doc = clean_text(doc.page_content)
        relevances.append(1 if clean_keyword in clean_doc else 0)

    dcg = calculate_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0

def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    """تقييم الاسترجاع لسؤال واحد"""
    retrieved_docs = fetch_context(test.question)
    
    if not test.keywords:
        return RetrievalEval(mrr=0.0, ndcg=0.0, keywords_found=0, total_keywords=0, keyword_coverage=0.0)

    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    ndcg_scores = [calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0

    return RetrievalEval(
        mrr=avg_mrr, ndcg=avg_ndcg, keywords_found=keywords_found, 
        total_keywords=total_keywords, keyword_coverage=keyword_coverage
    )


def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    """تقييم الإجابة لسؤال واحد (Sync)"""
    generated_answer, retrieved_docs = answer_question(test.question)

    judge_messages = [
        {
            "role": "system",
            "content": "أنت خبير في تقييم جودة إجابات أنظمة الاسترجاع (RAG) باللغة العربية. قم بتقييم الإجابة المولدة بمقارنتها بالإجابة النموذجية المرجعية بناءً على الدقة، الشمولية، والارتباط.",
        },
        {
            "role": "user",
            "content": f"""السؤال:
{test.question}

الإجابة المولدة من النظام:
{generated_answer}

الإجابة النموذجية المرجعية:
{test.reference_answer}

الرجاء تقييم الإجابة المولدة بناءً على 3 معايير من 1 (سيء جداً) إلى 5 (مثالي):
1. الدقة (Accuracy): هل المعلومات صحيحة بناءً على الإجابة النموذجية؟
2. الشمولية (Completeness): هل تمت تغطية جميع جوانب الإجابة النموذجية؟
3. الارتباط (Relevance): هل الإجابة مباشرة وتخص السؤال فقط دون إسهاب غير مبرر؟""",
        },
    ]

    try:
        judge_response = completion(
            model="ollama/" + JUDGE_MODEL, 
            messages=judge_messages, 
            response_format=AnswerEval
        )
        answer_eval = AnswerEval.model_validate_json(judge_response.choices[0].message.content)
    except Exception as e:
        print(f"Error evaluating test ID {test.id}: {e}")
        answer_eval = AnswerEval(feedback=f"Error: {e}", accuracy=1.0, completeness=1.0, relevance=1.0)

    return answer_eval, generated_answer, retrieved_docs


# ----------------------------------------------------
# الدوال التي يحتاجها Evaluator.py لربط واجهة Gradio
# ----------------------------------------------------



def evaluate_all_retrieval():
    """Generator لدعم شريط التقدم في Gradio لتقييم الاسترجاع"""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test)
        progress = (index + 1) / total_tests
        yield test, result, progress


def evaluate_all_answers():
    """Generator لدعم شريط التقدم في Gradio لتقييم الإجابات"""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result, _, _ = evaluate_answer(test)
        progress = (index + 1) / total_tests
        yield test, result, progress

def evaluate_all_retrieval():
    """Generator لدعم شريط التقدم في Gradio بشكل آمن ومستقر"""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        try:
            result = evaluate_retrieval(test)
            progress = (index + 1) / total_tests
            yield test, result, progress
        except Exception as e:
            print(f"Error in test ID {test.id}: {e}")
            yield test, RetrievalEval(mrr=0.0, ndcg=0.0, keywords_found=0, total_keywords=len(test.keywords), keyword_coverage=0.0), (index + 1) / total_tests


def evaluate_all_answers():
    """Generator لدعم شريط التقدم في Gradio لتقييم الإجابات بشكل آمن ومستقر"""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        try:
            result, _, _ = evaluate_answer(test)
            progress = (index + 1) / total_tests
            yield test, result, progress
        except Exception as e:
            print(f"Error evaluating answer for test ID {test.id}: {e}")
            err_eval = AnswerEval(feedback=f"Error: {e}", accuracy=1.0, completeness=1.0, relevance=1.0)
            yield test, err_eval, (index + 1) / total_tests











# ----------------------------------------------------
# تشغيل السكربت من الـ CLI
# ----------------------------------------------------

def run_cli_evaluation(test_number: int):
    tests = load_tests()

    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    test = tests[test_number]

    print(f"\n{'=' * 80}\nTest #{test_number} (ID: {test.id})")
    print(f"Question: {test.question}\nKeywords: {test.keywords}\nReference: {test.reference_answer}")

    print(f"\n{'=' * 80}\nRetrieval Evaluation")
    retrieved_res = evaluate_retrieval(test)
    print(f"MRR: {retrieved_res.mrr:.4f} | nDCG: {retrieved_res.ndcg:.4f} | Coverage: {retrieved_res.keyword_coverage:.1f}%")

    print(f"\n{'=' * 80}\nAnswer Evaluation")
    ans_res, gen_ans, _ = evaluate_answer(test)
    print(f"Generated Answer: {gen_ans}")
    print(f"Scores -> Accuracy: {ans_res.accuracy}/5 | Completeness: {ans_res.completeness}/5 | Relevance: {ans_res.relevance}/5")
    print(f"Feedback: {ans_res.feedback}\n{'=' * 80}\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: uv run Eval.py <test_row_number>")
        sys.exit(1)

    try:
        test_number = int(sys.argv[1])
    except ValueError:
        print("Error: test_row_number must be an integer")
        sys.exit(1)

    run_cli_evaluation(test_number)


if __name__ == "__main__":
    main()