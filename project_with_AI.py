from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    ForeignKey,
    Date
)

from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    Session,
    relationship
)

from datetime import date

import os
from dotenv import load_dotenv

from google import genai
from google.genai import types


# =========================================================
# GEMINI CONFIGURATION
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set in .env"
    )

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Student Exam Management System"
)


# =========================================================
# DATABASE
# =========================================================

DATABASE_URL = "sqlite:///./education.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False
    }
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False
)

Base = declarative_base()


def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


# =========================================================
# DATABASE MODELS
# =========================================================


class StudentDB(Base):

    __tablename__ = "students"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    name = Column(
        String,
        nullable=False
    )

    age = Column(Integer)

    email = Column(
        String,
        unique=True
    )

    enrollments = relationship(
        "EnrollmentDB",
        back_populates="student"
    )


class CourseDB(Base):

    __tablename__ = "courses"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    name = Column(
        String,
        nullable=False
    )

    description = Column(String)

    enrollments = relationship(
        "EnrollmentDB",
        back_populates="course"
    )

    exams = relationship(
        "ExamDB",
        back_populates="course"
    )


class EnrollmentDB(Base):

    __tablename__ = "enrollments"

    id = Column(
        Integer,
        primary_key=True
    )

    student_id = Column(
        Integer,
        ForeignKey("students.id")
    )

    course_id = Column(
        Integer,
        ForeignKey("courses.id")
    )

    student = relationship(
        "StudentDB",
        back_populates="enrollments"
    )

    course = relationship(
        "CourseDB",
        back_populates="enrollments"
    )


class ExamDB(Base):

    __tablename__ = "exams"

    id = Column(
        Integer,
        primary_key=True
    )

    title = Column(String)

    exam_date = Column(Date)

    course_id = Column(
        Integer,
        ForeignKey("courses.id")
    )

    course = relationship(
        "CourseDB",
        back_populates="exams"
    )

    questions = relationship(
        "QuestionDB",
        back_populates="exam"
    )


class QuestionDB(Base):

    __tablename__ = "questions"

    id = Column(
        Integer,
        primary_key=True
    )

    question_text = Column(String)

    exam_id = Column(
        Integer,
        ForeignKey("exams.id")
    )

    exam = relationship(
        "ExamDB",
        back_populates="questions"
    )

    options = relationship(
        "OptionDB",
        back_populates="question"
    )


class OptionDB(Base):

    __tablename__ = "options"

    id = Column(
        Integer,
        primary_key=True
    )

    option_text = Column(String)

    is_correct = Column(
        Boolean,
        default=False
    )

    question_id = Column(
        Integer,
        ForeignKey("questions.id")
    )

    question = relationship(
        "QuestionDB",
        back_populates="options"
    )


class ExamAttemptDB(Base):

    __tablename__ = "exam_attempts"

    id = Column(
        Integer,
        primary_key=True
    )

    student_id = Column(
        Integer,
        ForeignKey("students.id")
    )

    exam_id = Column(
        Integer,
        ForeignKey("exams.id")
    )

    score = Column(
        Integer,
        default=0
    )

    total_questions = Column(
        Integer,
        default=0
    )


# =========================================================
# CREATE TABLES
# =========================================================

Base.metadata.create_all(
    bind=engine
)


# =========================================================
# PYDANTIC MODELS
# =========================================================


class StudentCreate(BaseModel):

    name: str
    age: int
    email: str
    # email: EmailStr


class CourseCreate(BaseModel):

    name: str
    description: str


class EnrollmentCreate(BaseModel):

    student_id: int
    course_id: int


class ExamCreate(BaseModel):

    title: str

    exam_date: date

    course_id: int


class QuestionCreate(BaseModel):

    question_text: str

    exam_id: int


class OptionCreate(BaseModel):

    option_text: str

    is_correct: bool


class Answer(BaseModel):

    question_id: int

    option_id: int


class SubmitExam(BaseModel):

    student_id: int

    answers: list[Answer]


# =========================================================
# GEMINI PERFORMANCE ANALYSIS
# =========================================================


def generate_performance_feedback(
    student_name,
    exam_title,
    score,
    total_questions,
    percentage
):

    instruction = """
You are an AI educational performance analyzer.

Your job is to analyze a student's exam performance.

Follow these rules:

1. Analyze the student's performance based on the score
   and percentage provided.

2. Identify the student's performance level.

3. Explain the student's result in simple language.

4. Identify strengths based on the result.

5. Identify areas that need improvement.

6. Give practical study recommendations.

7. Give an encouraging message.

8. Keep the response clear, concise and student-friendly.

9. Do not invent information that is not provided.

10. Do not claim that the student is weak in a specific
    topic unless information about that topic is provided.
"""

    prompt = f"""
Analyze the following student's exam performance.

Student Name:
{student_name}

Exam:
{exam_title}

Score:
{score}/{total_questions}

Percentage:
{percentage:.2f}%

Provide the response using exactly these sections:

Performance Level:

Analysis:

Strengths:

Areas for Improvement:

Study Recommendations:

Encouraging Message:
"""

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=instruction
            )
        )

        print("================================")
        print("GEMINI RESPONSE OBJECT:")
        print(response)
        print("================================")

        # Get the generated text
        if response.text:
            return response.text

        return "Gemini returned an empty response."

    except Exception as e:

        print("================================")
        print("GEMINI ERROR TYPE:", type(e).__name__)
        print("GEMINI ERROR:", repr(e))
        print("================================")

        return "AI performance analysis is currently unavailable."

        # return "AI performance analysis is currently unavailable."


# =========================================================
# HOME
# =========================================================


@app.get("/")
def home():

    return {
        "message": "Student Exam Management API is running"
    }


# =========================================================
# STUDENT APIs
# =========================================================


@app.post("/students")
def create_student(
    student: StudentCreate,
    db: Session = Depends(get_db)
):

    existing = db.query(StudentDB).filter(
        StudentDB.email == student.email
    ).first()

    if existing:

        raise HTTPException(
            status_code=400,
            detail="Email already exists"
        )

    data = StudentDB(
        name=student.name,
        age=student.age,
        email=student.email
    )

    db.add(data)

    db.commit()

    db.refresh(data)

    return {
        "message": "Student created successfully",
        "student": {
            "id": data.id,
            "name": data.name,
            "age": data.age,
            "email": data.email
        }
    }


@app.get("/students")
def get_students(
    db: Session = Depends(get_db)
):

    return db.query(StudentDB).all()


@app.get("/students/{student_id}")
def get_student(
    student_id: int,
    db: Session = Depends(get_db)
):

    student = db.query(StudentDB).filter(
        StudentDB.id == student_id
    ).first()

    if not student:

        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    return student


@app.put("/students/{student_id}")
def update_student(
    student_id: int,
    student: StudentCreate,
    db: Session = Depends(get_db)
):

    data = db.query(StudentDB).filter(
        StudentDB.id == student_id
    ).first()

    if not data:

        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    data.name = student.name
    data.age = student.age
    data.email = student.email

    db.commit()

    db.refresh(data)

    return {
        "message": "Student updated successfully",
        "student": data
    }


@app.delete("/students/{student_id}")
def delete_student(
    student_id: int,
    db: Session = Depends(get_db)
):

    data = db.query(StudentDB).filter(
        StudentDB.id == student_id
    ).first()

    if not data:

        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    db.delete(data)

    db.commit()

    return {
        "message": "Student deleted successfully"
    }


# =========================================================
# COURSE APIs
# =========================================================


@app.post("/courses")
def create_course(
    course: CourseCreate,
    db: Session = Depends(get_db)
):

    data = CourseDB(
        name=course.name,
        description=course.description
    )

    db.add(data)

    db.commit()

    db.refresh(data)

    return {
        "message": "Course created successfully",
        "course": data
    }


@app.get("/courses")
def get_courses(
    db: Session = Depends(get_db)
):

    return db.query(CourseDB).all()


@app.get("/courses/{course_id}")
def get_course(
    course_id: int,
    db: Session = Depends(get_db)
):

    course = db.query(CourseDB).filter(
        CourseDB.id == course_id
    ).first()

    if not course:

        raise HTTPException(
            status_code=404,
            detail="Course not found"
        )

    return course


@app.put("/courses/{course_id}")
def update_course(
    course_id: int,
    course: CourseCreate,
    db: Session = Depends(get_db)
):

    data = db.query(CourseDB).filter(
        CourseDB.id == course_id
    ).first()

    if not data:

        raise HTTPException(
            status_code=404,
            detail="Course not found"
        )

    data.name = course.name

    data.description = course.description

    db.commit()

    db.refresh(data)

    return {
        "message": "Course updated successfully",
        "course": data
    }


@app.delete("/courses/{course_id}")
def delete_course(
    course_id: int,
    db: Session = Depends(get_db)
):

    data = db.query(CourseDB).filter(
        CourseDB.id == course_id
    ).first()

    if not data:

        raise HTTPException(
            status_code=404,
            detail="Course not found"
        )

    db.delete(data)

    db.commit()

    return {
        "message": "Course deleted successfully"
    }


# =========================================================
# ENROLLMENT APIs
# =========================================================


@app.post("/enroll")
def enroll_student(
    enrollment: EnrollmentCreate,
    db: Session = Depends(get_db)
):

    student = db.query(StudentDB).filter(
        StudentDB.id == enrollment.student_id
    ).first()

    if not student:

        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    course = db.query(CourseDB).filter(\
        CourseDB.id == enrollment.course_id
    ).first()

    if not course:

        raise HTTPException(
            status_code=404,
            detail="Course not found"
        )

    existing = db.query(EnrollmentDB).filter(
        EnrollmentDB.student_id == enrollment.student_id,
        EnrollmentDB.course_id == enrollment.course_id
    ).first()

    if existing:

        raise HTTPException(
            status_code=400,
            detail="Student already enrolled"
        )

    data = EnrollmentDB(
        student_id=enrollment.student_id,
        course_id=enrollment.course_id
    )

    db.add(data)

    db.commit()

    db.refresh(data)

    return {
        "message": "Student enrolled successfully"
    }


@app.delete("/enroll")
def de_enroll_student(
    enrollment: EnrollmentCreate,
    db: Session = Depends(get_db)
):

    data = db.query(EnrollmentDB).filter(
        EnrollmentDB.student_id == enrollment.student_id,
        EnrollmentDB.course_id == enrollment.course_id
    ).first()

    if not data:

        raise HTTPException(
            status_code=404,
            detail="Enrollment not found"
        )

    db.delete(data)

    db.commit()

    return {
        "message": "Student de-enrolled successfully"
    }


# =========================================================
# EXAM APIs
# =========================================================


@app.post("/exams")
def create_exam(
    exam: ExamCreate,
    db: Session = Depends(get_db)
):

    course = db.query(CourseDB).filter(
        CourseDB.id == exam.course_id
    ).first()

    if not course:

        raise HTTPException(
            status_code=404,
            detail="Course not found"
        )

    data = ExamDB(
        title=exam.title,
        exam_date=exam.exam_date,
        course_id=exam.course_id
    )

    db.add(data)

    db.commit()

    db.refresh(data)

    return {
        "message": "Exam created successfully",
        "exam": {
            "id": data.id,
            "title": data.title,
            "exam_date": data.exam_date,
            "course_id": data.course_id
        }
    }


# =========================================================
# QUESTION API
# =========================================================


@app.post("/questions")
def create_question(
    question: QuestionCreate,
    db: Session = Depends(get_db)
):

    exam = db.query(ExamDB).filter(
        ExamDB.id == question.exam_id
    ).first()

    if not exam:

        raise HTTPException(
            status_code=404,
            detail="Exam not found"
        )

    data = QuestionDB(
        question_text=question.question_text,
        exam_id=question.exam_id
    )

    db.add(data)

    db.commit()

    db.refresh(data)

    return {
        "message": "Question created successfully",
        "question": {
            "id": data.id,
            "question_text": data.question_text,
            "exam_id": data.exam_id
        }
    }


# =========================================================
# OPTION API
# =========================================================


@app.post("/questions/{question_id}/options")
def create_option(
    question_id: int,
    option: OptionCreate,
    db: Session = Depends(get_db)
):

    question = db.query(QuestionDB).filter(
        QuestionDB.id == question_id
    ).first()

    if not question:

        raise HTTPException(
            status_code=404,
            detail="Question not found"
        )

    data = OptionDB(
        option_text=option.option_text,
        is_correct=option.is_correct,
        question_id=question_id
    )

    db.add(data)

    db.commit()

    db.refresh(data)

    return {
        "message": "Option created successfully",
        "option": {
            "id": data.id,
            "option_text": data.option_text,
            "is_correct": data.is_correct,
            "question_id": data.question_id
        }
    }


# =========================================================
# LIST QUESTIONS
# =========================================================


@app.get("/exams/{exam_id}/questions")
def list_questions(
    exam_id: int,
    db: Session = Depends(get_db)
):

    exam = db.query(ExamDB).filter(
        ExamDB.id == exam_id
    ).first()

    if not exam:

        raise HTTPException(
            status_code=404,
            detail="Exam not found"
        )

    questions = db.query(QuestionDB).filter(
        QuestionDB.exam_id == exam_id
    ).all()

    result = []

    for question in questions:

        options = db.query(OptionDB).filter(
            OptionDB.question_id == question.id
        ).all()

        result.append({

            "question_id": question.id,

            "question": question.question_text,

            "options": [

                {
                    "option_id": option.id,
                    "option_text": option.option_text
                }

                for option in options
            ]
        })

    return {

        "exam_id": exam_id,

        "exam": exam.title,

        "exam_date": exam.exam_date,

        "questions": result
    }


# =========================================================
# SUBMIT EXAM
# =========================================================


@app.post("/exams/{exam_id}/submit")
def submit_exam(
    exam_id: int,
    submission: SubmitExam,
    db: Session = Depends(get_db)
):

    exam = db.query(ExamDB).filter(
        ExamDB.id == exam_id
    ).first()

    if not exam:

        raise HTTPException(
            status_code=404, 
            detail="Exam not found"
        )

    student = db.query(StudentDB).filter(
        StudentDB.id == submission.student_id
    ).first()

    if not student:

        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    questions = db.query(QuestionDB).filter(
        QuestionDB.exam_id == exam_id
    ).all()

    total_questions = len(questions)

    score = 0

    for answer in submission.answers:

        option = db.query(OptionDB).filter(

            OptionDB.id == answer.option_id,

            OptionDB.question_id == answer.question_id

        ).first()

        if option and option.is_correct:

            score += 1

    attempt = ExamAttemptDB(

        student_id=submission.student_id,

        exam_id=exam_id,

        score=score,

        total_questions=total_questions
    )

    db.add(attempt)

    db.commit()

    db.refresh(attempt)

    percentage = 0

    if total_questions > 0:

        percentage = (
            score /
            total_questions
        ) * 100

    return {

        "message": "Exam submitted successfully",

        "student_id": submission.student_id,

        "exam_id": exam_id,

        "exam": exam.title,

        "exam_date": exam.exam_date,

        "score": score,

        "total_questions": total_questions,

        "percentage": percentage
    }


# =========================================================
# PERFORMANCE REPORT
# =========================================================


@app.get("/reports/student/{student_id}")
def performance_report(
    student_id: int,
    db: Session = Depends(get_db)
):

    student = db.query(StudentDB).filter(
        StudentDB.id == student_id
    ).first()

    if not student:

        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    attempts = db.query(ExamAttemptDB).filter(
        ExamAttemptDB.student_id == student_id
    ).all()

    reports = []

    total_score = 0

    total_questions = 0

    for attempt in attempts:

        exam = db.query(ExamDB).filter(
            ExamDB.id == attempt.exam_id
        ).first()

        if not exam:

            continue

        percentage = 0

        if attempt.total_questions > 0:

            percentage = (
                attempt.score /
                attempt.total_questions
            ) * 100


        # =================================================
        # GEMINI AI ANALYSIS
        # =================================================

        ai_feedback = generate_performance_feedback(

            student_name=student.name,

            exam_title=exam.title,

            score=attempt.score,

            total_questions=attempt.total_questions,

            percentage=percentage
        )


        # =================================================
        # ADD EXAM REPORT
        # =================================================

        reports.append({

            "exam_id": attempt.exam_id,

            "exam": exam.title,

            "exam_date": exam.exam_date,

            "score": attempt.score,

            "total_questions": attempt.total_questions,

            "percentage": percentage,

            "ai_feedback": ai_feedback
        })


        total_score += attempt.score

        total_questions += attempt.total_questions


    # =====================================================
    # OVERALL PERCENTAGE
    # =====================================================

    overall_percentage = 0

    if total_questions > 0:

        overall_percentage = (
            total_score /
            total_questions
        ) * 100


    return {

        "student_id": student.id,

        "student_name": student.name,

        "exams": reports,

        "overall_score": total_score,

        "overall_questions": total_questions,

        "overall_percentage": overall_percentage
    }