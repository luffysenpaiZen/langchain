from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str='issac' #default
    age: Optional[int]=None  #Optional
    email:EmailStr="none@gmail.com"
    cgpa:float=Field(gt=0,lt=10,default=5,description='this is float representation of cgpa of the student')
    

new_student={'age':'32','email':'issac.chowdary@zenshastra.com'}

student = Student(**new_student) # type: ignore

student_dict = dict(student)

print(student_dict['age'])

student_json = student.model_dump_json()

print(student_json)


