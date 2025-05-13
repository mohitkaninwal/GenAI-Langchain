from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name: str
    age:Optional[int]
    email:EmailStr
    cgpa: float= Field(gt=0,lt=10)

new_student = {'name':'Mohit', 'age':23, 'email':'mohit37.kumar@nickelfox.com','cgpa':2}

student = Student(**new_student)

student_dict=dict(student)
print(student_dict['age'])