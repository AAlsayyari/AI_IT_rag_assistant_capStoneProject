# import json
# import os
# from pathlib import Path
# from pydantic import BaseModel, Field

# # تعديل اسم الملف ليتطابق مع الملف الذي أنشأناه
# TEST_FILE = str(Path(__file__).parent / "tests.jsonl")


# class TestQuestion(BaseModel):
#     """نموذج بيانات السؤال المتوافق مع ملف jsonl الخاص بمشروعنا."""
    
#     id: int = Field(description="المعرف الفريد للسؤال")
#     question: str = Field(description="السؤال المراد توجيهه لنظام RAG")
#     keywords: list[str] = Field(default_factory=list, description="الكلمات المفتاحية التي يجب ظهورها في الاسترجاع")
#     reference_answer: str = Field(description="الإجابة النموذجية المرجعية للتقييم")
#     category: str = Field(default="", description="تصنيف السؤال (مثل: direct_fact, workflow_procedure)")
#     context_source: str = Field(default="", description="اسم الملف المرجعي أو المصدر")


# def load_tests() -> list[TestQuestion]:
#     """قراءة أسئلة التقييم من ملف JSONL."""
#     tests = []
    
#     if not os.path.exists(TEST_FILE):
#         raise FileNotFoundError(f"Error: Dataset file not found at {TEST_FILE}")

#     with open(TEST_FILE, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
                
#             data = json.loads(line)
            
#             # فلترة ذكية: نأخذ فقط الأوبجكت التي تحتوي على سؤال وإجابة مرجعية 
#             # (هذا يحمي النظام من التعطل إذا كان الملف يحتوي على أسطر fine-tuning مختلفة الهيكل)
#             if "question" in data and "reference_answer" in data:
#                 tests.append(TestQuestion(**data))
                
#     return tests






import json
import os
from pathlib import Path
from pydantic import BaseModel, Field

# تحديد مكان ملف الـ jsonl
TEST_FILE = str(Path(__file__).parent / "tests.jsonl")


class TestQuestion(BaseModel):
    """نموذج بيانات السؤال المتوافق مع ملف jsonl الخاص بمشروعنا."""
    
    id: int = Field(alias="test_id", description="المعرف الفريد للسؤال")
    question: str = Field(description="السؤال المراد توجيهه لنظام RAG")
    keywords: list[str] = Field(default_factory=list, description="الكلمات المفتاحية التي يجب ظهورها في الاسترجاع")
    reference_answer: str = Field(alias="ground_truth", description="الإجابة النموذجية المرجعية للتقييم")
    category: str = Field(default="", description="تصنيف السؤال")
    context_source: str = Field(default="", alias="source_file", description="اسم الملف المرجعي أو المصدر")

    class Config:
        populate_by_name = True


def load_tests() -> list[TestQuestion]:
    """قراءة أسئلة التقييم من ملف JSONL."""
    tests = []
    
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Error: Dataset file not found at {TEST_FILE}")

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            data = json.loads(line)
            
            # التأكد من وجود السؤال والإجابة بغض النظر عن اسم المفتاح
            if "question" in data and ("reference_answer" in data or "ground_truth" in data):
                tests.append(TestQuestion.model_validate(data))
                
    return tests