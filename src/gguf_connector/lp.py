
filename = "server.py"
content = """
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from livepoll.database import Poll, Option, create_tables, get_db
app = FastAPI(title="Live Poll API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
create_tables()
class CreatePollRequest(BaseModel):
    question: str
    options: List[str]
class OptionOut(BaseModel):
    id: int
    text: str
    votes: int
    class Config:
        from_attributes = True
class PollOut(BaseModel):
    id: str
    question: str
    is_active: bool
    options: List[OptionOut]
    class Config:
        from_attributes = True
class VoteRequest(BaseModel):
    option_id: int
@app.post("/polls", response_model=PollOut)
def create_poll(body: CreatePollRequest, db: Session = Depends(get_db)):
    if len(body.options) < 2:
        raise HTTPException(status_code=400, detail="At least 2 options required.")
    poll = Poll(question=body.question, is_active=True)
    db.add(poll)
    db.flush()
    for opt_text in body.options:
        if opt_text.strip():
            db.add(Option(text=opt_text.strip(), poll_id=poll.id))
    db.commit()
    db.refresh(poll)
    return poll
@app.get("/polls/{poll_id}", response_model=PollOut)
def get_poll(poll_id: str, db: Session = Depends(get_db)):
    poll = db.query(Poll).filter(Poll.id == poll_id).first()
    if not poll:
        raise HTTPException(status_code=404, detail="Poll not found.")
    return poll
@app.post("/polls/{poll_id}/vote", response_model=PollOut)
def vote(poll_id: str, body: VoteRequest, db: Session = Depends(get_db)):
    poll = db.query(Poll).filter(Poll.id == poll_id).first()
    if not poll:
        raise HTTPException(status_code=404, detail="Poll not found.")
    if not poll.is_active:
        raise HTTPException(status_code=400, detail="Poll has ended.")
    option = db.query(Option).filter(Option.id == body.option_id, Option.poll_id == poll_id).first()
    if not option:
        raise HTTPException(status_code=404, detail="Option not found.")
    option.votes += 1
    db.commit()
    db.refresh(poll)
    return poll
@app.get("/polls/{poll_id}/results", response_model=PollOut)
def get_results(poll_id: str, db: Session = Depends(get_db)):
    poll = db.query(Poll).filter(Poll.id == poll_id).first()
    if not poll:
        raise HTTPException(status_code=404, detail="Poll not found.")
    return poll
@app.post("/polls/{poll_id}/end", response_model=PollOut)
def end_poll(poll_id: str, db: Session = Depends(get_db)):
    poll = db.query(Poll).filter(Poll.id == poll_id).first()
    if not poll:
        raise HTTPException(status_code=404, detail="Poll not found.")
    poll.is_active = False
    db.commit()
    db.refresh(poll)
    return poll
"""

with open(filename, "w") as f:
    f.write(content)
    
import os
os.system("uvicorn server:app --reload --host 0.0.0.0 --port 8000")
