from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Numbers(BaseModel):
    num1: int
    num2: int

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/sum2nums")
async def sum_two_numbers(num1: int, num2: int):
    return {"sum": num1 + num2}

@app.get("/mult2nums")
async def mult_two_numbers(num1: int, num2: int):
    return {"multiplication": num1 * num2}

@app.post("/mult2nums/")
async def mult_two_numbers_post(numbers: Numbers):
    return {"multiplication": numbers.num1 * numbers.num2}

@app.post("/sum2nums/")
async def mult_two_numbers_post(numbers: Numbers):
    return {"sum": numbers.num1 + numbers.num2}
